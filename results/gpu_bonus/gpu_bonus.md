# Bonus: DROO on CPU vs GPU

Adaptive K (Delta = 32), seed 0, 3000 frames per run. GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU.

| N | Device | Total s/frame | Network s/frame | Solver + rest s/frame | Avg normalized rate |
|---|---|---|---|---|---|
| 10 | CPU | 0.0212 | 0.00078 | 0.0204 | 0.99370 |
| 10 | CUDA | 0.0204 | 0.00195 | 0.0184 | 0.99349 |
| 30 | CPU | 0.1885 | 0.00107 | 0.1875 | 0.95261 |
| 30 | CUDA | 0.2058 | 0.00295 | 0.2029 | 0.95750 |

GPU vs CPU:

- N = 10: network part 0.40x, whole run 1.04x (above 1 = GPU faster, below 1 = GPU slower)
- N = 30: network part 0.36x, whole run 0.92x (above 1 = GPU faster, below 1 = GPU slower)
