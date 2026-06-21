#!/usr/bin/env python3
"""Launch the self-hosted QLLM dashboard.

    python -m qllm.dashboard.run                 # serve built UI on :8000
    python -m qllm.dashboard.run --port 9000
    python -m qllm.dashboard.run --db results/qllm_results.db

Then open http://localhost:8000. For live frontend development with hot
reload, instead run the API here and `npm run dev` in dashboard/frontend
(Vite proxies /api to :8000).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default="results/qllm_results.db")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    os.environ["QLLM_DB"] = args.db
    os.environ["QLLM_RESULTS"] = args.results
    dist = Path(__file__).parent / "frontend" / "dist"
    if not dist.exists():
        print(
            "WARNING: frontend not built. Run: "
            "cd qllm/dashboard/frontend && npm install && npm run build"
        )
        print("   (serving API only for now)")

    import uvicorn

    print(f"QLLM dashboard on http://{args.host}:{args.port}  (db: {args.db})")
    uvicorn.run("qllm.dashboard.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
