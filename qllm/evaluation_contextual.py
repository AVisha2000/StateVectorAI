"""Constrained-position accuracy: the metric that exposes the memory wall.

On the contextual-parity task, only the parity-FORCED tokens (mask==1)
separate models: predicting them requires recalling every earlier
observable value in their context. A memoryless predictor scores chance
(0.5) there; a model with enough memory scores ~1.0. We report accuracy on
constrained vs unconstrained positions separately — the gap is the signal.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def constrained_accuracy(model, params, ids: np.ndarray, mask: np.ndarray,
                         seq_len: int = 64, n_windows: int = 64,
                         seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    mask = np.asarray(mask)

    @jax.jit
    def logits_fn(batch):
        return model.apply({"params": params}, batch)

    con_hit = con_tot = unc_hit = unc_tot = 0
    starts = rng.integers(0, len(ids) - seq_len - 1, size=n_windows)
    for st in starts:
        window = ids[st:st + seq_len][None, :]
        target = ids[st + 1:st + seq_len + 1]
        tmask = mask[st + 1:st + seq_len + 1]
        pred = np.asarray(logits_fn(jnp.asarray(window))[0]).argmax(-1)
        hit = (pred == target)
        con = tmask == 1
        con_hit += int(hit[con].sum()); con_tot += int(con.sum())
        unc_hit += int(hit[~con].sum()); unc_tot += int((~con).sum())

    return {
        "constrained_acc": con_hit / max(con_tot, 1),
        "unconstrained_acc": unc_hit / max(unc_tot, 1),
        "separation": con_hit / max(con_tot, 1) - 0.5,  # vs chance on parity bit
    }
