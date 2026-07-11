"""Regenerate or verify the committed QLLM Dashboard OpenAPI contract."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "qllm" / "dashboard" / "openapi.json"


def _openapi_document() -> dict:
    """Load the app against an isolated database so contract dumps are read-only."""
    with tempfile.TemporaryDirectory(prefix="qllm-openapi-") as temp_dir:
        temp = Path(temp_dir)
        os.environ["QLLM_DB"] = str(temp / "openapi.db")
        os.environ["QLLM_RESULTS"] = str(temp / "results")
        os.environ["QLLM_DATA"] = str(temp / "data")
        os.environ["QLLM_DISABLE_WORKER"] = "1"
        sys.path.insert(0, str(ROOT))

        from qllm.dashboard import server

        try:
            return server.app.openapi()
        finally:
            server.QUEUE.close()


def render_openapi() -> str:
    return json.dumps(_openapi_document(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the committed snapshot is stale",
    )
    args = parser.parse_args(argv)
    rendered = render_openapi()

    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                "OpenAPI snapshot is stale; run: python scripts/dump_openapi.py",
                file=sys.stderr,
            )
            return 1
        print(f"OpenAPI snapshot is current: {OUTPUT.relative_to(ROOT)}")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
