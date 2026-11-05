# Bonus: DROO on CPU vs GPU

Adaptive K (Delta = 32), seed 0, 3000 frames per run. GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU.

| N | Device | Total s/frame | Network s/frame | Solver + rest s/frame | Avg normalized rate |
|---|---|---|---|---|---|
| 10 | CPU | 0.0069 | 0.00022 | 0.0066 | 0.99370 |
| 10 | CUDA | 0.0077 | 0.00122 | 0.0065 | 0.99349 |
| 30 | CPU | 0.0649 | 0.00045 | 0.0644 | 0.95261 |
| 30 | CUDA | 0.0710 | 0.00172 | 0.0692 | 0.95750 |

GPU vs CPU:

- N = 10: network part 0.18x, whole run 0.89x (above 1 = GPU faster, below 1 = GPU slower)
- N = 30: network part 0.26x, whole run 0.91x (above 1 = GPU faster, below 1 = GPU slower)
