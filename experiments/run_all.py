"""
run_all.py - runs every remaining experiment of the paper in parallel,
then draws all figures and tables.

    python experiments/run_all.py                # use the default number of workers
    python experiments/run_all.py --workers 8    # choose how many run at once
    python experiments/run_all.py --plot-only    # just redraw from saved results
    python experiments/run_all.py --dry-run      # list the jobs, run nothing
    python experiments/run_all.py --quick        # tiny test of everything
                                                 # (writes to quicktest/ only)

Safe to stop and restart: a job whose results already exist is skipped.
Each job writes its own log to results/logs/<experiment>__<job>.log.

Covered here: Fig. 4, 7, 9, 10, 11, 12, 13, 14, 15 and Table 2.
(Fig. 5, 6 and 8 have their own scripts and were already run.)
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if "--quick" in sys.argv:          # must be set before importing common
    os.environ["DROO_QUICK"] = "1"

import fig04            # noqa: E402
import fig07            # noqa: E402
import fig09            # noqa: E402
import fig10            # noqa: E402
import fig11_13         # noqa: E402
import table2_fig14_15  # noqa: E402
from common import RESULTS_DIR  # noqa: E402

# (module, script file, experiment folder). Order = start order: longest first.
MODULES = [
    (table2_fig14_15, "table2_fig14_15.py", table2_fig14_15.EXPERIMENT),
    (fig07, "fig07.py", fig07.EXPERIMENT),
    (fig09, "fig09.py", fig09.EXPERIMENT),
    (fig10, "fig10.py", fig10.EXPERIMENT),
    (fig11_13, "fig11_13.py", fig11_13.EXPERIMENT),
]
PLOTS = ["fig04.py", "fig07.py", "fig09.py", "fig10.py", "fig11_13.py",
         "table2_fig14_15.py"]


def all_jobs():
    jobs = []
    for module, script, experiment in MODULES:
        for job in module.JOBS:
            done = RESULTS_DIR / experiment / (module.job_tag(job) + "_summary.json")
            jobs.append((script, job, experiment, done))
    return jobs


def run_one(script, job, experiment):
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / ("%s__%s.log" % (experiment, job.replace("=", "_")))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    # (DROO_QUICK, if set, is inherited from os.environ)
    start = time.time()
    with open(log, "w") as f:
        code = subprocess.call([sys.executable, str(HERE / script), "--run", job],
                               stdout=f, stderr=subprocess.STDOUT, env=env)
    return code, time.time() - start, log


def main():
    parser = argparse.ArgumentParser(description="Run all remaining experiments")
    cores = os.cpu_count() or 2
    default_workers = max(1, cores // 2 - 1)   # roughly physical cores minus one
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="parallel jobs (default %d)" % default_workers)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="tiny frame counts, results in quicktest/")
    args = parser.parse_args()

    jobs = all_jobs()
    todo = [j for j in jobs if not j[3].exists()]
    print("%d jobs in total, %d already done, %d to run, %d at a time"
          % (len(jobs), len(jobs) - len(todo), len(todo), args.workers))

    if args.dry_run:
        for script, job, _, done in jobs:
            print("  %-22s %-20s %s" % (script, job, "done" if done.exists() else "to run"))
        return

    failed = []
    if not args.plot_only and todo:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, s, j, e): (s, j) for s, j, e, _ in todo}
            for k, fut in enumerate(as_completed(futures), 1):
                script, job = futures[fut]
                code, secs, log = fut.result()
                status = "ok" if code == 0 else "FAILED (see %s)" % log
                if code != 0:
                    failed.append((script, job, log))
                print("[%2d/%d] %6.0f s  %-22s %-20s %s   (elapsed %.0f min)"
                      % (k, len(todo), secs, script, job, status,
                         (time.time() - t0) / 60), flush=True)

    if failed:
        print("\n%d job(s) failed. Fix them, then run this script again "
              "(finished jobs are skipped):" % len(failed))
        for script, job, log in failed:
            print("   %s --run %s   log: %s" % (script, job, log))
        return

    print("\nDrawing figures and tables...")
    for script in PLOTS:
        print("\n=== %s ===" % script, flush=True)
        subprocess.call([sys.executable, str(HERE / script), "--plot"])
    where = "quicktest/" if os.environ.get("DROO_QUICK") else "figures/ and results/"
    print("\nAll done. Output is in %s." % where)


if __name__ == "__main__":
    main()
