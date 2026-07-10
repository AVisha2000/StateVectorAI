# QLLM Verification Matrix

The selector in `scripts/verify_changes.py` is the executable source of truth. This table explains its intent.

| Changed area | Minimum safe evidence |
|---|---|
| `.agents/`, `.codex/`, `AGENTS.md`, `PLANS.md` | `python scripts/check_agent_setup.py` and agent-configuration tests |
| `qllm/dashboard/frontend/` | `npm run build` from the frontend directory |
| `qllm/dashboard/` backend or API | `pytest -q tests/test_dashboard_lab.py`; queue smoke when queue/API behavior changes |
| `qllm/`, `benchmarks/`, `scripts/`, `configs/`, or `tests/` Python behavior | focused tests first; full `pytest -q` for shared contracts or broad blast radius |
| docs only | static agent/setup validation; no runtime test unless docs encode behavior |
| `RESULTS.md`, claim/evidence docs, studies reports, evidence code | research-protocol review plus human gate |
| `docs/RESEARCH_PROGRAM.md`, `docs/RESEARCH_MAP.yaml` planning/map updates | research-protocol review when claims are affected; not an automatic human gate |
| GPU/QPU configs, providers, long runs, publishing or git delivery | human gate; never auto-execute from verification |

On Windows, use a unique pytest temp root if the system temp directory raises `WinError 5`, for example `pytest -q --basetemp .tmp/pytest-$PID` in PowerShell.
