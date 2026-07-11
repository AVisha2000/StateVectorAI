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

from .security import configure_access


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default="results/qllm_results.db")
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit non-loopback clients (exposes local research data)",
    )
    ap.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="explicit browser origin allowed in remote mode; repeat as needed",
    )
    args = ap.parse_args()

    try:
        configure_access(
            host=args.host,
            allow_remote=args.allow_remote,
            cors_origins=args.cors_origin,
        )
    except ValueError as exc:
        ap.error(str(exc))

    os.environ["QLLM_DB"] = args.db
    os.environ["QLLM_RESULTS"] = args.results
    os.environ["QLLM_DATA"] = args.data
    dist = Path(__file__).parent / "frontend" / "dist"
    if not dist.exists():
        print(
            "WARNING: frontend not built. Run: "
            "cd qllm/dashboard/frontend && npm install && npm run build"
        )
        print("   (serving API only for now)")

    import uvicorn

    if args.allow_remote:
        print("!" * 72)
        print("WARNING: REMOTE ACCESS ENABLED")
        print("This dashboard can expose local datasets, runs, and environment data.")
        print(f"Allowed browser origins: {', '.join(args.cors_origin)}")
        print("!" * 72)
    print(f"QLLM dashboard on http://{args.host}:{args.port}  (db: {args.db})")
    uvicorn.run("qllm.dashboard.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
