# Table 2 replication: CPU execution latency per channel (seconds)

Ours = CPU time on this computer; paper = Intel i5-4570 (in brackets).

| N | DROO fixed K=N | DROO adaptive (Delta=32) | CD | LR (solver reused) | LR (rebuilt per channel) |
|---|---|---|---|---|---|
| 10 | 1.1e-02 (3.6e-2) | 4.8e-03 (1.2e-2) | 6.2e-02 (2.0e-1) | 2.5e-03 (2.4e-1) | 4.4e-02 |
| 20 | 4.3e-02 (1.3e-1) | 1.1e-02 (3.0e-2) | 4.0e-01 (1.3) | 3.9e-03 (5.3e-1) | 9.6e-02 |
| 30 | 1.1e-01 (3.1e-1) | 1.7e-02 (5.9e-2) | 1.2e+00 (3.8) | 5.2e-03 (8.1e-1) | 1.3e-01 |

How many times slower than DROO (adaptive K) - ours (paper):

| N | DROO fixed K / adaptive | CD / DROO | LR reused / DROO | LR rebuilt / DROO |
|---|---|---|---|---|
| 10 | 2.4x (3.0x) | 13x (17x) | 0.5x (20x) | 9.1x (20x) |
| 20 | 4.0x (4.3x) | 37x (43x) | 0.4x (18x) | 8.8x (18x) |
| 30 | 6.4x (5.3x) | 74x (64x) | 0.3x (14x) | 7.9x (14x) |

# Fig. 14 numbers: average computation rate (bits/s) on the 6,000 test channels

| N | CD | DROO | LR | Edge | Local |
|---|---|---|---|---|---|
| 10 | 3.186e+06 | 3.185e+06 | 3.16e+06 | 2.891e+06 | 9.42e+05 |
| 20 | 4.372e+06 | 4.354e+06 | 4.299e+06 | 3.687e+06 | 1.887e+06 |
| 30 | 6.025e+06 | 5.973e+06 | 5.9e+06 | 4.856e+06 | 2.856e+06 |

# Fig. 15 numbers: normalized rate on the 6,000 test channels

| N | DROO median | DROO mean | DROO 25th pct | DROO 5th pct | DROO >= 0.99 | DROO > 1 | LR median | LR mean | LR 25th pct | LR >= 0.99 | CD (100 ch.) mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1.0000 | 0.9997 | 1.0000 | 0.9983 | 99.2% | 0.1% | 0.9924 | 0.9915 | 0.9864 | 60.9% | 1.0000 |
| 20 | 1.0000 | 0.9953 | 0.9947 | 0.9801 | 86.4% | 0.0% | 0.9840 | 0.9831 | 0.9765 | 28.4% | 1.0000 |
| 30 | 0.9980 | 0.9902 | 0.9898 | 0.9542 | 74.7% | 0.1% | 0.9796 | 0.9789 | 0.9724 | 14.7% | 1.0000 |
