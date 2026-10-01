"""Connect 4: GPU Monte-Carlo player (X) vs GPU heuristic player (O). Run: python gpu_connect4.py [playouts_per_column]"""
import sys, numpy as np
from numba import cuda

R, C, M = 6, 7, 0xFFFFFFFF

@cuda.jit(device=True)
def drop(b, c, p):                       # place piece p in column c, return row or -1
    for r in range(R - 1, -1, -1):
        if b[r * C + c] == 0:
            b[r * C + c] = p
            return r
    return -1

@cuda.jit(device=True)
def wins(b, r, c, p):                    # 4-in-a-row through (r,c)?
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < R and 0 <= cc < C and b[rr * C + cc] == p:
                n += 1; rr += s * dr; cc += s * dc
        if n >= 4:
            return True
    return False

# ---- Player X kernel: one thread = one random playout (7*n threads in parallel) ----
@cuda.jit
def mc_kernel(board, p, n, seed, score):
    i = cuda.grid(1)
    if i >= C * n: return
    b = cuda.local.array(42, np.int8)
    col = i // n
    for k in range(R * C): b[k] = board[k]
    r = drop(b, col, p)
    if r < 0: return
    if wins(b, r, col, p):
        cuda.atomic.add(score, col, 2); return
    s, cur, res = (seed + i * 2654435761) % M | 1, 3 - p, 1           # res=1 -> draw
    for _ in range(R * C):
        s ^= (s << 13) & M; s ^= s >> 17; s ^= (s << 5) & M            # xorshift32
        c = s % C
        for k in range(C):
            if b[(c + k) % C] == 0: c = (c + k) % C; break
        r = drop(b, c, cur)
        if r < 0: break
        if wins(b, r, c, cur):
            res = 2 if cur == p else 0; break
        cur = 3 - cur
    cuda.atomic.add(score, col, res)

# ---- Player O kernel: one thread = one column, 1-ply win/block/centre heuristic ----
@cuda.jit
def heur_kernel(board, p, score):
    c = cuda.grid(1)
    if c >= C: return
    b = cuda.local.array(42, np.int8)
    for k in range(R * C): b[k] = board[k]
    r = drop(b, c, p)
    if r < 0: score[c] = -1; return
    s = 3 - abs(c - 3)                                                 # prefer centre
    if wins(b, r, c, p): s += 1000
    else:
        b[r * C + c] = 0
        r = drop(b, c, 3 - p)
        if wins(b, r, c, 3 - p): s += 500                              # must-block
    score[c] = s

def show(b):
    print("\n".join("|" + " ".join(".XO"[v] for v in b[r*C:(r+1)*C]) + "|" for r in range(R)) + "\n 0 1 2 3 4 5 6\n")

def play(n):
    board, turn = np.zeros(R * C, np.int8), 1
    for t in range(R * C):
        sc = np.zeros(C, np.int64)
        if turn == 1: mc_kernel[(C * n + 127) // 128, 128](board, 1, n, 1234 + t * 977, sc)
        else:         heur_kernel[1, 32](board, 2, sc)
        sc[board[:C] != 0] = -1                                        # full columns illegal
        c = int(sc.argmax())
        r = 5 - int((board[c::C] != 0).sum())
        board[r * C + c] = turn
        print(f"Move {t+1}: {'X (GPU Monte-Carlo)' if turn == 1 else 'O (GPU heuristic)'} -> col {c}  scores={sc.tolist()}")
        show(board)
        if check(board, r, c, turn):
            print(f"{'X' if turn == 1 else 'O'} wins!"); return
        turn = 3 - turn
    print("Draw")

def check(b, r, c, p):
    g = b.reshape(R, C)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < R and 0 <= cc < C and g[rr, cc] == p: n += 1; rr += s * dr; cc += s * dc
        if n >= 4: return True
    return False

if __name__ == "__main__":
    play(int(sys.argv[1]) if len(sys.argv) > 1 else 2048)