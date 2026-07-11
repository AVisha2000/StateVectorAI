#!/usr/bin/env python3
"""Queue a tiny dashboard-launched run without using the browser.

Useful after starting the dashboard server:
    python scripts/queue_smoke.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--preset", default="classical-small")
    ap.add_argument("--dataset", default="default-text")
    ap.add_argument("--run-name", default="dashboard-smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device-target", choices=["auto", "cpu", "gpu"], default="cpu")
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--compare", action="store_true",
                    help="Queue the preset's classical twin when available.")
    args = ap.parse_args()

    payload = {
        "preset_id": args.preset,
        "dataset_name": args.dataset,
        "run_name": args.run_name,
        "seed": args.seed,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "device_target": args.device_target,
        "queue_classical_comparison": args.compare,
        "checkpoint_every": args.checkpoint_every,
        "resume_from": args.resume_from,
    }
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/api/jobs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        job = json.loads(resp.read().decode("utf-8"))
    print(json.dumps(job, indent=2))
    deadline = time.monotonic() + args.timeout
    terminal = {"done", "error", "cancelled"}
    while job.get("status") not in terminal:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Job {job.get('id')} did not finish within {args.timeout}s"
            )
        time.sleep(0.5)
        with urllib.request.urlopen(
            f"{args.url.rstrip('/')}/api/jobs/{job['id']}", timeout=10
        ) as resp:
            job = json.loads(resp.read().decode("utf-8"))
    print(json.dumps({
        "id": job.get("id"),
        "status": job.get("status"),
        "completed_step": job.get("completed_step"),
        "checkpoint_path": job.get("checkpoint_path"),
        "recovery_count": job.get("recovery_count"),
        "error": job.get("error"),
    }, indent=2))
    if job.get("status") != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
