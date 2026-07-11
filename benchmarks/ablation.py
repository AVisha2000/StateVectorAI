#!/usr/bin/env python3
"""2x2 ablation: isolate the quantum circuit's contribution.

Grid (planning-doc design; Quixer-style parameter matching):
  quantum-trained   : VQC FFN, circuit weights trained
  quantum-frozen    : VQC FFN, circuit weights FROZEN at random init
  classical-matched : classical FFN, d_ff solved so total params match
  classical-full    : reference classical model from the base config

Decision rules:
  - quantum-trained must beat quantum-frozen, else circuit training adds nothing
  - quantum-trained must beat classical-matched, else no quantum contribution

Runs are tagged by config stem, results accumulate in
results/ablation_<tag>.csv, and the report regenerates from all rows —
so the grid can be filled across multiple invocations via --only.

Usage:
    python benchmarks/ablation.py --base-config configs/quantum_ffn_sharp.yaml \
        --only quantum-trained quantum-frozen
    python benchmarks/ablation.py --base-config configs/quantum_ffn_sharp.yaml \
        --only classical-matched classical-full
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import inspect
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import load_yaml  # noqa: E402
from qllm.data.datasets import data_config_hash  # noqa: E402
from qllm.data.text import CharTokenizer, load_corpus  # noqa: E402
from qllm.models.model import matched_classical_d_ff  # noqa: E402
from qllm.research_protocol import (  # noqa: E402
    classify_claim,
    normalize_seed_axes,
    paired_improvements,
    paired_power_plan,
    paired_stats,
    practical_equivalence,
)
from qllm.train.loop import fit  # noqa: E402

ALL_VARIANTS = (
    "quantum-trained",
    "quantum-frozen",
    "classical-matched",
    "classical-full",
)
FIELDS = [
    "variant",
    "seed",
    "n_params",
    "val_loss",
    "val_ppl",
    "wall_seconds",
    "grad_norm_ratio",
    "protocol_version",
    "protocol_hash",
    "data_config_hash",
    "data_kind",
    "data_gen_seed",
    "steps",
    "eval_every",
    "device_target",
    "batch_size",
    "seq_len",
    "lr",
    "weight_decay",
    "grad_clip",
    "eval_batches",
    "seed_axes_json",
]

DEFAULT_EQUIVALENCE_MARGIN = 0.10
DEFAULT_SMALLEST_USEFUL_EFFECT = 0.10


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "variant": r["variant"],
                    "seed": int(r["seed"]),
                    "n_params": int(r["n_params"]),
                    "val_loss": float(r["val_loss"]),
                    "val_ppl": float(r["val_ppl"]),
                    "wall_seconds": float(r["wall_seconds"]),
                    "grad_norm_ratio": (
                        float(r["grad_norm_ratio"])
                        if r.get("grad_norm_ratio") not in (None, "", "None")
                        else None
                    ),
                    "protocol_version": (
                        int(r["protocol_version"])
                        if r.get("protocol_version") not in (None, "")
                        else None
                    ),
                    "protocol_hash": r.get("protocol_hash") or None,
                    "data_config_hash": r.get("data_config_hash") or None,
                    "data_kind": r.get("data_kind") or None,
                    "data_gen_seed": (
                        int(r["data_gen_seed"])
                        if r.get("data_gen_seed") not in (None, "")
                        else None
                    ),
                    "steps": int(r["steps"]) if r.get("steps") else None,
                    "eval_every": (
                        int(r["eval_every"]) if r.get("eval_every") else None
                    ),
                    "device_target": r.get("device_target") or None,
                    "batch_size": (
                        int(r["batch_size"]) if r.get("batch_size") else None
                    ),
                    "seq_len": int(r["seq_len"]) if r.get("seq_len") else None,
                    "lr": float(r["lr"]) if r.get("lr") else None,
                    "weight_decay": (
                        float(r["weight_decay"])
                        if r.get("weight_decay") else None
                    ),
                    "grad_clip": (
                        float(r["grad_clip"]) if r.get("grad_clip") else None
                    ),
                    "eval_batches": (
                        int(r["eval_batches"])
                        if r.get("eval_batches") else None
                    ),
                    "seed_axes_json": r.get("seed_axes_json") or None,
                }
            )
    return rows


def _protocol_metadata(cfg, *, circuit_applicable: bool) -> dict[str, Any]:
    """Persist the protocol needed to verify rows accumulated across invocations."""
    import jax

    protocol = {
        "protocol_version": 2,
        "data_config_hash": data_config_hash(cfg.data),
        "data_kind": cfg.data.kind,
        "data_gen_seed": cfg.data.gen_seed,
        "steps": cfg.train.steps,
        "eval_every": cfg.train.eval_every,
        "device_target": jax.default_backend(),
        "batch_size": cfg.train.batch_size,
        "seq_len": cfg.train.seq_len,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "grad_clip": cfg.train.grad_clip,
        "eval_batches": cfg.train.eval_batches,
    }
    identity = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    axes = normalize_seed_axes(
        cfg.train.seed,
        generator_seed=cfg.data.gen_seed,
        data_kind=cfg.data.kind,
        circuit_applicable=circuit_applicable,
    )
    return {
        **protocol,
        "protocol_hash": hashlib.sha256(identity).hexdigest(),
        "seed_axes_json": json.dumps(
            axes, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    }


def _as_payload(value: Any) -> dict[str, Any]:
    """Normalize protocol dataclasses and dictionaries for report rendering."""
    if isinstance(value, dict):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"unsupported research-protocol payload: {type(value)!r}")


def _classify_paired_result(
    *,
    fairness: dict[str, bool],
    stats: Any,
    equivalence: dict[str, Any],
    power: dict[str, Any],
    analogue_ladder: dict[str, Any],
) -> dict[str, Any]:
    """Use every claim-classification input supported by the installed API."""
    kwargs: dict[str, Any] = {
        "fairness": fairness,
        "paired": stats,
        "min_pairs": 6,
        "metric_name": "validation perplexity",
    }
    supported = inspect.signature(classify_claim).parameters
    if "equivalence" in supported:
        kwargs["equivalence"] = equivalence
    if "power" in supported:
        kwargs["power"] = power
    if "analogue_ladder" in supported:
        kwargs["analogue_ladder"] = analogue_ladder
    return classify_claim(**kwargs)


def _rows_by_seed(rows: list[dict], variant: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for row in rows:
        if row["variant"] != variant:
            continue
        seed = int(row["seed"])
        if seed in indexed:
            raise ValueError(f"duplicate ablation row for {variant!r}, seed {seed}")
        indexed[seed] = row
    return indexed


_PROTOCOL_FIELDS = (
    "protocol_version",
    "protocol_hash",
    "data_config_hash",
    "data_kind",
    "data_gen_seed",
    "steps",
    "eval_every",
    "device_target",
    "batch_size",
    "seq_len",
    "lr",
    "weight_decay",
    "grad_clip",
    "eval_batches",
    "seed_axes_json",
)


def _decoded_seed_axes(row: dict) -> dict[str, Any] | None:
    raw = row.get("seed_axes_json")
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _seed_axes_match_row(row: dict) -> bool:
    axes = _decoded_seed_axes(row)
    if axes is None or not axes.get("supported"):
        return False
    seed = int(row["seed"])
    if axes.get("initialization") != seed or axes.get("minibatch") != seed:
        return False
    expected_generator = (
        int(row["data_gen_seed"])
        if row.get("data_kind") not in (None, "text")
        and row.get("data_gen_seed") is not None
        else None
    )
    if axes.get("generator") != expected_generator:
        return False
    expected_circuit = seed if str(row["variant"]).startswith("quantum-") else None
    return axes.get("circuit") == expected_circuit


def _ablation_fairness(
    candidate_by_seed: dict[int, dict],
    baseline_by_seed: dict[int, dict],
    paired_seeds: list[int],
    *,
    complete_pairs: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paired_rows = [
        row
        for seed in paired_seeds
        for row in (candidate_by_seed[seed], baseline_by_seed[seed])
    ]
    complete_metadata = bool(paired_rows) and all(
        all(row.get(field) is not None for field in _PROTOCOL_FIELDS)
        for row in paired_rows
    )

    def one_non_null_value(field: str) -> bool:
        values = {row.get(field) for row in paired_rows}
        return len(values) == 1 and None not in values

    same_protocol = one_non_null_value("protocol_hash")
    seed_axes_valid = complete_metadata and all(
        _seed_axes_match_row(row) for row in paired_rows
    )
    fairness = {
        "same_dataset": one_non_null_value("data_config_hash"),
        "same_seed": complete_pairs,
        "same_steps": one_non_null_value("steps"),
        "same_eval_interval": one_non_null_value("eval_every"),
        "same_device_target": one_non_null_value("device_target"),
        "same_training_budget": all(
            one_non_null_value(field)
            for field in (
                "batch_size", "seq_len", "lr", "weight_decay",
                "grad_clip", "eval_batches",
            )
        ),
        "same_preprocessing": one_non_null_value("data_config_hash"),
        "role_validation": complete_pairs and complete_metadata,
        "protocol_complete": complete_metadata,
        "same_protocol_hash": same_protocol,
        "seed_axes_valid": seed_axes_valid,
    }
    fairness["valid"] = all(bool(value) for value in fairness.values())
    mismatches = [
        {
            "path": key,
            "reason": (
                "legacy row is missing protocol provenance"
                if key == "protocol_complete"
                else "accumulated ablation rows do not share this protocol field"
            ),
            "allowed": False,
        }
        for key, value in fairness.items()
        if key != "valid" and not value
    ]
    return fairness, mismatches


def _ablation_analogue_ladder(
    rows: list[dict], candidate: str, baseline: str, paired_seeds: list[int]
) -> dict[str, Any]:
    frozen_seeds = {
        int(row["seed"]) for row in rows if row["variant"] == "quantum-frozen"
    }
    classical_seeds = {
        int(row["seed"])
        for row in rows
        if row["variant"] in {"classical-matched", "classical-full"}
    }
    frozen_complete = set(paired_seeds) <= frozen_seeds
    classical_complete = set(paired_seeds) <= classical_seeds
    missing = ["resource_accounting"]
    if not frozen_complete:
        missing.append("frozen_random_control")
    if not classical_complete:
        missing.append("strong_classical_challenger")
    return {
        "required_complete": False,
        "missing_required": missing,
        "rungs": [
            {
                "id": "frozen_random_control",
                "required": True,
                "status": "met" if frozen_complete else "unknown",
            },
            {
                "id": "strong_classical_challenger",
                "required": True,
                "status": "met" if classical_complete else "unknown",
            },
            {
                "id": "resource_accounting",
                "required": True,
                "status": "unknown",
                "detail": (
                    "legacy ablation rows do not record circuit calls; wall time "
                    "and parameters alone cannot complete the resource rung"
                ),
            },
        ],
        "candidate": candidate,
        "baseline": baseline,
    }


def paired_ablation_analysis(
    rows: list[dict],
    candidate: str,
    baseline: str,
    *,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    smallest_useful_effect: float = DEFAULT_SMALLEST_USEFUL_EFFECT,
) -> dict[str, Any]:
    """Analyze a comparison only across exact seed-matched result rows.

    The result deliberately retains unmatched seed lists.  A partial seed
    intersection can still be inspected, but it fails the fairness gate and
    cannot support an empirical-edge label.
    """
    if equivalence_margin <= 0.0:
        raise ValueError("equivalence_margin must be > 0")
    if smallest_useful_effect <= 0.0:
        raise ValueError("smallest_useful_effect must be > 0")

    candidate_by_seed = _rows_by_seed(rows, candidate)
    baseline_by_seed = _rows_by_seed(rows, baseline)
    candidate_seeds = set(candidate_by_seed)
    baseline_seeds = set(baseline_by_seed)
    paired_seeds = sorted(candidate_seeds & baseline_seeds)
    complete_pairs = bool(paired_seeds) and candidate_seeds == baseline_seeds

    fairness, fairness_mismatches = _ablation_fairness(
        candidate_by_seed,
        baseline_by_seed,
        paired_seeds,
        complete_pairs=complete_pairs,
    )
    analogue_ladder = _ablation_analogue_ladder(
        rows, candidate, baseline, paired_seeds
    )
    result: dict[str, Any] = {
        "candidate": candidate,
        "baseline": baseline,
        "paired_seeds": paired_seeds,
        "unmatched_candidate_seeds": sorted(candidate_seeds - baseline_seeds),
        "unmatched_baseline_seeds": sorted(baseline_seeds - candidate_seeds),
        "fairness": fairness,
        "fairness_mismatches": fairness_mismatches,
        "analogue_ladder": analogue_ladder,
        "paired_stats": None,
        "equivalence": None,
        "power": None,
    }

    if paired_seeds:
        candidate_scores = [
            float(candidate_by_seed[seed]["val_ppl"]) for seed in paired_seeds
        ]
        baseline_scores = [
            float(baseline_by_seed[seed]["val_ppl"]) for seed in paired_seeds
        ]
        stats = paired_stats(
            candidate_scores,
            baseline_scores,
            lower_is_better=True,
            bootstrap_seed=0,
            sign_flip_seed=0,
        )
        equivalence = practical_equivalence(stats, margin=equivalence_margin)
        power = paired_power_plan(
            paired_improvements(
                candidate_scores,
                baseline_scores,
                lower_is_better=True,
            ),
            smallest_useful_effect=smallest_useful_effect,
        )
        result["paired_stats"] = _as_payload(stats)
        result["equivalence"] = _as_payload(equivalence)
        result["power"] = _as_payload(power)
        result["claim"] = _classify_paired_result(
            fairness=fairness,
            stats=stats,
            equivalence=equivalence,
            power=power,
            analogue_ladder=analogue_ladder,
        )
    else:
        result["claim"] = classify_claim(
            fairness=fairness,
            metric_name="validation perplexity",
        )
    return result


def _aggregate_variant(rows: list[dict], name: str) -> tuple[float, float, int, list[dict]]:
    variant_rows = sorted(
        (row for row in rows if row["variant"] == name),
        key=lambda row: int(row["seed"]),
    )
    ppls = [float(row["val_ppl"]) for row in variant_rows]
    mean = statistics.mean(ppls)
    std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
    return mean, std, int(variant_rows[0]["n_params"]), variant_rows


def _format_pair_analysis(title: str, analysis: dict[str, Any]) -> list[str]:
    seeds = analysis["paired_seeds"]
    claim = analysis["claim"]
    if not seeds:
        return [
            f"- {title}: no shared seeds; verdict: {claim['label']} — "
            f"{claim['reason']}."
        ]

    stats = analysis["paired_stats"]
    equivalence = analysis["equivalence"]
    power = analysis["power"]
    ci_method = stats.get("ci_method", "paired interval")
    sign_flip_method = stats.get("sign_flip_method", "sign flip")
    seed_text = ", ".join(str(seed) for seed in seeds)
    lines = [
        f"- {title}: paired n={stats['n_pairs']} on seeds [{seed_text}]; "
        f"mean Δppl={stats['mean_improvement']:+.3f}, "
        f"95% {ci_method} CI [{stats['ci_low']:+.3f}, "
        f"{stats['ci_high']:+.3f}], {sign_flip_method} "
        f"p={stats['p_value']:.4f}."
    ]
    if analysis["unmatched_candidate_seeds"] or analysis["unmatched_baseline_seeds"]:
        lines.append(
            "  - unmatched rows: candidate-only seeds "
            f"{analysis['unmatched_candidate_seeds']}; baseline-only seeds "
            f"{analysis['unmatched_baseline_seeds']}."
        )
    if analysis.get("fairness_mismatches"):
        lines.append(
            "  - protocol fairness failed: "
            + "; ".join(
                f"{item['path']} ({item['reason']})"
                for item in analysis["fairness_mismatches"]
            )
            + "."
        )
    else:
        lines.append(
            "  - seed lineage: generator and deterministic split are recorded; "
            "initialization/minibatch and applicable circuit axes share the "
            "paired train seed."
        )

    equivalent = equivalence.get("equivalent")
    eq_label = (
        "equivalent"
        if equivalent is True
        else "not equivalent"
        if equivalent is False
        else equivalence.get("status", "not assessed")
    )
    lines.append(
        f"  - practical equivalence: {eq_label} at ±"
        f"{float(equivalence['margin']):.3f} ppl."
    )

    recommended = power.get("recommended_pairs")
    recommendation = (
        "unavailable"
        if recommended is None
        else f"{int(recommended)} paired seeds"
    )
    lines.append(
        f"  - power planning: {power.get('status', 'unavailable')}; "
        f"recommended total={recommendation}; adequately powered="
        f"{bool(power.get('adequately_powered', False))}."
    )
    lines.append(f"  - verdict: {claim['label']} — {claim['reason']}.")
    return lines


def build_ablation_report(
    rows: list[dict],
    *,
    tag: str,
    steps: int,
    base_config: str,
    matched_d_ff: int,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    smallest_useful_effect: float = DEFAULT_SMALLEST_USEFUL_EFFECT,
) -> str:
    """Render the accumulated grid using paired, conservative inference."""
    have = {row["variant"] for row in rows}
    lines = [
        f"# 2x2 ablation results — {tag}",
        "",
        f"{steps} steps, base={base_config}, matched d_ff={matched_d_ff}",
        "",
        "| variant | params | val ppl (mean ± std) | per-seed |",
        "|---|---|---|---|",
    ]
    for name in ALL_VARIANTS:
        if name not in have:
            continue
        mean, std, n_params, variant_rows = _aggregate_variant(rows, name)
        per_seed = ", ".join(
            f"s{int(row['seed'])}={float(row['val_ppl']):.2f}"
            for row in variant_rows
        )
        lines.append(
            f"| {name} | {n_params:,} | {mean:.3f} ± {std:.3f} | {per_seed} |"
        )

    if have >= set(ALL_VARIANTS):
        lines += [
            "",
            "## Paired decision rules",
            "",
            "Positive Δppl means the first-listed candidate has lower "
            "perplexity. Inference uses exact seed pairs; three pairs remain "
            "pilot evidence regardless of effect direction.",
            "",
        ]
        trained_frozen = paired_ablation_analysis(
            rows,
            "quantum-trained",
            "quantum-frozen",
            equivalence_margin=equivalence_margin,
            smallest_useful_effect=smallest_useful_effect,
        )
        trained_classical = paired_ablation_analysis(
            rows,
            "quantum-trained",
            "classical-matched",
            equivalence_margin=equivalence_margin,
            smallest_useful_effect=smallest_useful_effect,
        )
        lines.extend(
            _format_pair_analysis("trained vs frozen circuit", trained_frozen)
        )
        lines.extend(
            _format_pair_analysis(
                "trained quantum vs parameter-matched classical",
                trained_classical,
            )
        )
    else:
        missing = sorted(set(ALL_VARIANTS) - have)
        lines += ["", f"(partial grid — missing: {', '.join(missing)})"]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/quantum_ffn_4q.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--only", nargs="+", choices=ALL_VARIANTS, default=None)
    parser.add_argument("--tag", default=None, help="default: config file stem")
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=DEFAULT_EQUIVALENCE_MARGIN,
        help="practical-equivalence margin in validation perplexity",
    )
    parser.add_argument(
        "--smallest-useful-effect",
        type=float,
        default=DEFAULT_SMALLEST_USEFUL_EFFECT,
        help="smallest useful validation-perplexity improvement for power planning",
    )
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    assert base.model.ffn_type == "quantum", "base config must use a quantum FFN"
    tag = args.tag or Path(args.base_config).stem

    vocab = CharTokenizer(load_corpus(base.data.corpus_path)).vocab_size
    mq = base.model
    d_ff_matched = matched_classical_d_ff(mq, vocab)
    print(f"[{tag}] parameter-matched classical twin: d_ff={d_ff_matched}")

    variants = {
        "quantum-trained": mq,
        "quantum-frozen": dataclasses.replace(
            mq, quantum=dataclasses.replace(mq.quantum, trainable=False)
        ),
        "classical-matched": dataclasses.replace(
            mq, ffn_type="classical", d_ff=d_ff_matched
        ),
        "classical-full": dataclasses.replace(mq, ffn_type="classical"),
    }
    selected = args.only or list(ALL_VARIANTS)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"ablation_{tag}.csv"
    rows = [r for r in _load_existing(csv_path) if r["variant"] not in selected]

    t_start = time.time()
    for name in selected:
        model_cfg = variants[name]
        for seed in args.seeds:
            cfg = dataclasses.replace(
                base,
                model=model_cfg,
                train=dataclasses.replace(base.train, seed=seed, steps=args.steps),
                tracking=dataclasses.replace(
                    base.tracking,
                    experiment="qllm-ablation",
                    run_name=f"abl-{tag}-{name}-s{seed}",
                    log_quantum_diagnostics=False,
                ),
            )
            res = fit(cfg, verbose=False, out_dir=args.out)
            s = res["summary"]
            ratio = s["history"][-1].get("grad_norm_ratio")
            protocol_metadata = _protocol_metadata(
                cfg, circuit_applicable=name.startswith("quantum-")
            )
            rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "n_params": s["n_params"],
                    "val_loss": s["val_loss"],
                    "val_ppl": s["val_ppl"],
                    "wall_seconds": s["wall_seconds"],
                    "grad_norm_ratio": ratio,
                    **protocol_metadata,
                }
            )
            extra = f"  g_ratio={ratio:.2e}" if ratio is not None else ""
            print(
                f"{name:18s} seed={seed}  params={s['n_params']:7,d}  "
                f"val_ppl={s['val_ppl']:.3f}  ({s['wall_seconds']:.0f}s){extra}"
            )

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = build_ablation_report(
        rows,
        tag=tag,
        steps=args.steps,
        base_config=args.base_config,
        matched_d_ff=d_ff_matched,
        equivalence_margin=args.equivalence_margin,
        smallest_useful_effect=args.smallest_useful_effect,
    )
    (out / f"ablation_{tag}.md").write_text(report)
    print("\n" + report)
    print(f"wall: {time.time() - t_start:.0f}s -> {out / f'ablation_{tag}.md'}")


if __name__ == "__main__":
    main()
