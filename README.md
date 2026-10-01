# Competing GPUs: Connect 4

Two CUDA GPU kernels play Connect 4 against each other using different strategies.

| Player | Strategy | GPU mapping |
|---|---|---|
| X | Monte-Carlo playouts | 1 thread = 1 random full game, `7 x N` threads per move; results summed per column with `atomic.add` |
| O | 1-ply heuristic (win +1000, block +500, centre bonus) | 1 thread = 1 column |

The host loop alternates the two kernels on the same GPU (works on single-GPU setups like Coursera). After every move it prints the mover, chosen column, per-column scores and an ASCII board.

## Run
```bash
pip install -r requirements.txt
python gpu_connect4.py            # 2048 playouts per column
python gpu_connect4.py 8          # weaker X -> O can win
```
No NVIDIA GPU? Use the CUDA simulator (slow, for testing):
```bash
NUMBA_ENABLE_CUDASIM=1 python gpu_connect4.py 8      # Linux/macOS
set NUMBA_ENABLE_CUDASIM=1 && python gpu_connect4.py 8   # Windows cmd
```

## Files
- `gpu_connect4.py` - game, both kernels, console visualization
- `presentation_script.md` - script for the demo video
- `sample_output.txt` - example game output
