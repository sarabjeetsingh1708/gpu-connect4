# Presentation Script (~4 min) — Competing GPUs: Connect 4

## 1. Project Description (0:00–1:00)
- "I built Connect 4 where both players are GPU kernels with *different* strategies."
- Player X: **Monte-Carlo kernel** — 7 columns × N random playouts, all running in parallel (one thread = one full game).
- Player O: **Heuristic kernel** — one thread per column; scores win-now (+1000), must-block (+500), centre preference.
- Single Coursera GPU → the host loop alternates the two kernels on the same device (the thread + lock-file approach is an optional extension).

## 2. Code Description (1:00–2:45) — show gpu_connect4.py
- Board = flat `int8[42]` (0 empty, 1 = X, 2 = O), copied into each thread's local memory.
- `drop()` / `wins()` are shared `@cuda.jit(device=True)` functions.
- `mc_kernel`: thread `i` → column `i // n`; plays that column, then random moves (per-thread xorshift32 RNG, seeded by thread id) until the game ends; result (win 2 / draw 1 / loss 0) added to `score[col]` with `cuda.atomic.add`.
- Host picks `argmax(score)` over legal columns → that is X's move.
- `heur_kernel`: thread = column; simulates own move, checks win; undoes and simulates opponent move to detect a block.
- Complexity: MC = O(7·N·42) work, but depth-42 parallel time; heuristic = O(1) per thread.

## 3. Demonstration / Visualization (2:45–4:00)
- Run: `python gpu_connect4.py 2048`
- After every move the console prints: who moved, chosen column, the 7 column scores, and the ASCII board (`X`, `O`, `.`).
- Point at the score vector: MC scores are win-rates (higher = better), heuristic scores show 1000/500 spikes for win/block.
- Expected result: X (Monte-Carlo) usually wins; reduce playouts (e.g. `python gpu_connect4.py 8`) to see it play weaker and O win — shows the GPU parallelism/strength tradeoff.
