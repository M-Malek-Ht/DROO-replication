"""
gpu_benchmark.py - BONUS experiment (not in the paper): does a GPU make
DROO faster?

The authors' network (DROO/memoryPyTorch.py, MemoryDNN) always runs on the
CPU. Here, MemoryDNNDevice is a copy of it that can run on any PyTorch device
("cpu" or "cuda"). It is a subclass: the network, the training step and the
order-preserving quantization are the authors' own; we only move the model
and the data to the chosen device. The authors' files are not changed.

For each N and each device the same DROO run is made (adaptive K, Delta = 32,
seed 0), and we measure:
  - network time per frame: action generation (decode) + training (learn)
  - total time per frame (also includes the (P2) solver, which is plain
    Python/SciPy and always runs on the CPU)

Expectation: little or no gain. The network is tiny (two hidden layers,
200 neurons) and ~89% of the time is the CPU-only solver; copying data to
and from the GPU every frame can even cost more than it saves.

How to run (from the project folder, in the GPU environment):
    python experiments/gpu_benchmark.py                 # N = 10 and 30, 3,000 frames
    python experiments/gpu_benchmark.py --frames 300    # quick test
Results: results/gpu_bonus/summary.json and results/gpu_bonus/gpu_bonus.md
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import RESULTS_DIR, environment_info, run_droo
from memoryPyTorch import MemoryDNN          # the authors' class


class MemoryDNNDevice(MemoryDNN):
    """The authors' MemoryDNN, able to run on a GPU, with timers added."""

    device = torch.device("cpu")              # set before each run

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)     # builds the authors' network
        self.model.to(self.device)
        self.net_time = 0.0                   # seconds spent in the network

    def learn(self):
        # Same as the authors' learn(), with tensors placed on the device
        start = time.perf_counter()
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        h_train = torch.tensor(batch_memory[:, 0: self.net[0]],
                               dtype=torch.float32, device=self.device)
        m_train = torch.tensor(batch_memory[:, self.net[0]:],
                               dtype=torch.float32, device=self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                               betas=(0.09, 0.999), weight_decay=0.0001)
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()
        self.cost = loss.item()               # .item() waits for the GPU
        assert self.cost > 0
        self.cost_his.append(self.cost)
        self.net_time += time.perf_counter() - start

    def decode(self, h, k=1, mode="OP"):
        # Same as the authors' decode(), with the input placed on the device
        start = time.perf_counter()
        h = torch.tensor(h[np.newaxis, :], dtype=torch.float32, device=self.device)
        self.model.eval()
        m_pred = self.model(h).detach().cpu().numpy()   # .cpu() waits for the GPU
        result = self.knm(m_pred[0], k) if mode == "OP" else self.knn(m_pred[0], k)
        self.net_time += time.perf_counter() - start
        return result


def measure(N, device, frames):
    MemoryDNNDevice.device = torch.device(device)
    # Warm-up run (not measured): PyTorch's one-time start-up costs
    # (and, on the GPU, loading CUDA) would otherwise count as DROO's time.
    run_droo(N=N, n=50, K=N, delta=32, seed=0, verbose=False,
             memory_class=MemoryDNNDevice)
    print("\nN = %d on %s, %d frames" % (N, device.upper(), frames), flush=True)
    r = run_droo(N=N, n=frames, K=N, delta=32, seed=0,
                 memory_class=MemoryDNNDevice)
    mem = r["mem"]
    out = {"N": N, "device": device, "frames": frames,
           "total_per_frame_s": r["summary"]["time_per_frame_s"],
           "network_per_frame_s": mem.net_time / frames,
           "avg_normalized_rate_test": r["summary"]["avg_normalized_rate_test"]}
    out["solver_and_rest_per_frame_s"] = out["total_per_frame_s"] - out["network_per_frame_s"]
    print("  total %.4f s/frame | network %.5f s/frame | rate %.5f"
          % (out["total_per_frame_s"], out["network_per_frame_s"],
             out["avg_normalized_rate_test"]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Bonus: CPU vs GPU for DROO")
    parser.add_argument("--frames", type=int, default=3000)
    args = parser.parse_args()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        gpu = torch.cuda.get_device_name(0)
        print("GPU found:", gpu)
    else:
        gpu = None
        print("No GPU visible to PyTorch - only the CPU will be measured.")

    rows = [measure(N, d, args.frames) for N in (10, 30) for d in devices]

    folder = RESULTS_DIR / "gpu_bonus"
    folder.mkdir(parents=True, exist_ok=True)
    tag = "" if args.frames == 3000 else "_frames%d" % args.frames
    env = environment_info()
    env["gpu"] = gpu
    with open(folder / ("summary%s.json" % tag), "w") as f:
        json.dump({"rows": rows, "environment": env}, f, indent=2)

    lines = ["# Bonus: DROO on CPU vs GPU",
             "",
             "Adaptive K (Delta = 32), seed 0, %d frames per run. GPU: %s."
             % (args.frames, gpu or "none"),
             "",
             "| N | Device | Total s/frame | Network s/frame | Solver + rest s/frame | Avg normalized rate |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %d | %s | %.4f | %.5f | %.4f | %.5f |" % (
            r["N"], r["device"].upper(), r["total_per_frame_s"],
            r["network_per_frame_s"], r["solver_and_rest_per_frame_s"],
            r["avg_normalized_rate_test"]))
    if gpu:
        lines += ["", "GPU vs CPU:", ""]
        for N in (10, 30):
            c = next(r for r in rows if r["N"] == N and r["device"] == "cpu")
            g = next(r for r in rows if r["N"] == N and r["device"] == "cuda")
            lines.append("- N = %d: network part %.2fx, whole run %.2fx "
                         "(above 1 = GPU faster, below 1 = GPU slower)" % (
                             N, c["network_per_frame_s"] / g["network_per_frame_s"],
                             c["total_per_frame_s"] / g["total_per_frame_s"]))
    text = "\n".join(lines) + "\n"
    (folder / ("gpu_bonus%s.md" % tag)).write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
