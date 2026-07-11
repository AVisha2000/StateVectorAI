"""Canonical dataset bundles and dependency-light dataset dispatch.

``load_dataset_bundle`` preserves trajectory boundaries and auxiliary masks;
``load_dataset`` remains the compatibility adapter returning a flat token
stream and tokenizer.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from ..config import DataConfig
from ..registry import DATASET_KINDS, choices_text
from .quantum_seq import (
    IdentityTokenizer,
    markov_control_sequences,
    monitored_ising_sequences,
)
from .text import CharTokenizer, load_corpus

SamplerPolicy = Literal["contiguous_stream", "within_trajectory"]


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


def canonical_data_config_json(cfg: DataConfig) -> str:
    """Stable JSON identity for every data-generation input."""
    return json.dumps(
        dataclasses.asdict(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def data_config_hash(cfg: DataConfig) -> str:
    """SHA-256 of the full, unrounded canonical :class:`DataConfig`."""
    payload = canonical_data_config_json(cfg).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _update_array_hash(digest, name: str, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(arr.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(arr.tobytes(order="C"))


def _tokenizer_identity(tokenizer: object | None) -> dict[str, Any]:
    """Return a stable description of the token-to-symbol interpretation."""
    if tokenizer is None:
        return {"kind": "unspecified"}
    identity: dict[str, Any] = {
        "kind": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    }
    itos = getattr(tokenizer, "itos", None)
    if isinstance(itos, Mapping):
        identity["id_to_token"] = [
            [int(index), str(token)]
            for index, token in sorted(itos.items(), key=lambda item: int(item[0]))
        ]
    else:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if isinstance(vocab_size, (int, np.integer)):
            identity["vocab_size"] = int(vocab_size)
    return identity


def dataset_content_hash(
    ids: np.ndarray,
    masks: Mapping[str, np.ndarray] | None = None,
    *,
    tokenizer: object,
) -> str:
    """SHA-256 of token data, masks, and token-to-symbol interpretation."""
    digest = hashlib.sha256()
    _update_array_hash(digest, "ids", ids)
    for name, mask in sorted((masks or {}).items()):
        _update_array_hash(digest, f"mask:{name}", mask)
    digest.update(b"tokenizer\0")
    digest.update(
        json.dumps(
            _tokenizer_identity(tokenizer),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    """Immutable dataset identity with explicit sampling/boundary semantics."""

    ids: np.ndarray
    tokenizer: object
    sampler_policy: SamplerPolicy
    masks: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    config_hash: str
    content_hash: str

    def __post_init__(self) -> None:
        ids = np.array(self.ids, dtype=np.int32, order="C", copy=True)
        if ids.ndim not in (1, 2) or ids.size == 0:
            raise ValueError("DatasetBundle.ids must be a non-empty 1-D or 2-D array.")
        expected_policy = (
            "within_trajectory" if ids.ndim == 2 else "contiguous_stream"
        )
        if self.sampler_policy != expected_policy:
            raise ValueError(
                f"sampler_policy must be '{expected_policy}' for ids.ndim={ids.ndim}."
            )
        ids.setflags(write=False)

        normalized_masks: dict[str, np.ndarray] = {}
        for name, value in self.masks.items():
            if not isinstance(name, str) or not name:
                raise ValueError("DatasetBundle mask names must be non-empty strings.")
            mask = np.array(value, order="C", copy=True)
            if mask.shape != ids.shape:
                raise ValueError(
                    f"DatasetBundle mask '{name}' shape {mask.shape} does not "
                    f"match ids shape {ids.shape}."
                )
            mask.setflags(write=False)
            normalized_masks[name] = mask

        expected_content_hash = dataset_content_hash(
            ids, normalized_masks, tokenizer=self.tokenizer
        )
        if self.content_hash != expected_content_hash:
            raise ValueError(
                "DatasetBundle.content_hash does not match ids/masks/tokenizer."
            )
        if not all(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (self.config_hash, self.content_hash)
        ):
            raise ValueError("DatasetBundle hashes must be SHA-256 hex digests.")

        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "masks", MappingProxyType(normalized_masks))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.ids.shape

    @property
    def sequence_shape(self) -> tuple[int, int] | None:
        """Trajectory matrix shape, or ``None`` for a continuous text stream."""
        return self.ids.shape if self.is_trajectory_data else None

    @property
    def is_trajectory_data(self) -> bool:
        return self.sampler_policy == "within_trajectory"

    @property
    def n_tokens(self) -> int:
        return int(self.ids.size)

    @property
    def n_sequences(self) -> int:
        return int(self.ids.shape[0]) if self.is_trajectory_data else 1

    @property
    def sequence_length(self) -> int:
        return int(self.ids.shape[1]) if self.is_trajectory_data else int(self.ids.size)

    @property
    def boundary_offsets(self) -> tuple[int, ...]:
        """Flat offsets including both 0 and the final exclusive boundary."""
        if not self.is_trajectory_data:
            return (0, self.n_tokens)
        length = self.sequence_length
        return tuple(range(0, self.n_tokens + 1, length))

    @property
    def boundaries(self) -> tuple[int, ...]:
        """Compatibility alias for :attr:`boundary_offsets`."""
        return self.boundary_offsets

    @property
    def provenance(self) -> Mapping[str, Any]:
        value = self.metadata.get("provenance", {})
        return value if isinstance(value, Mapping) else MappingProxyType({})


def _bundle(
    cfg: DataConfig,
    ids: np.ndarray,
    tokenizer: object,
    *,
    masks: Mapping[str, np.ndarray] | None = None,
    provenance: Mapping[str, Any],
) -> DatasetBundle:
    ids = np.asarray(ids, dtype=np.int32)
    normalized_masks = {
        name: np.asarray(mask) for name, mask in (masks or {}).items()
    }
    policy: SamplerPolicy = (
        "within_trajectory" if ids.ndim == 2 else "contiguous_stream"
    )
    metadata = {
        "kind": cfg.kind,
        "config": dataclasses.asdict(cfg),
        "provenance": dict(provenance),
    }
    return DatasetBundle(
        ids=ids,
        tokenizer=tokenizer,
        sampler_policy=policy,
        masks=normalized_masks,
        metadata=metadata,
        config_hash=data_config_hash(cfg),
        content_hash=dataset_content_hash(
            ids, normalized_masks, tokenizer=tokenizer
        ),
    )


def _cache_paths(cfg: DataConfig, legacy_key: str) -> tuple[Path, Path]:
    cache_dir = Path("results/.data_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    current = cache_dir / f"{cfg.kind}_{data_config_hash(cfg)}.npz"
    return current, cache_dir / f"{legacy_key}.npz"


def _legacy_cache_identity_is_exact(cfg: DataConfig) -> bool:
    """Whether a rounded legacy key uniquely represents this configuration."""
    if cfg.kind == "seq_cancellation":
        return cfg.seq_cancel_density == round(cfg.seq_cancel_density, 3)
    if cfg.kind in {"monitored_ising", "markov_control"}:
        return (
            cfg.gen_theta_zz == round(cfg.gen_theta_zz, 4)
            and cfg.gen_theta_x == round(cfg.gen_theta_x, 4)
        )
    return True


def _cache_payload_is_valid(
    cfg: DataConfig, payload: Mapping[str, np.ndarray]
) -> bool:
    """Check cached arrays before giving them canonical bundle semantics."""
    if not {"ids", "vocab"} <= set(payload):
        return False
    ids = np.asarray(payload["ids"])
    vocab_array = np.asarray(payload["vocab"])
    if ids.size != cfg.gen_sequences * cfg.gen_len or vocab_array.size != 1:
        return False
    vocab = int(vocab_array.reshape(-1)[0])
    if vocab <= 0 or not np.issubdtype(ids.dtype, np.integer):
        return False
    if np.any(ids < 0) or np.any(ids >= vocab):
        return False
    if cfg.kind == "contextual":
        if "mask" not in payload:
            return False
        mask = np.asarray(payload["mask"])
        if mask.shape != ids.shape or not np.issubdtype(mask.dtype, np.integer):
            return False
        if np.any((mask != 0) & (mask != 1)):
            return False
    return True


def _stored_hash_state(
    payload: Mapping[str, np.ndarray], name: str, expected: str
) -> bool | None:
    """Return ``None`` for legacy absence, otherwise whether the hash matches."""
    if name not in payload:
        return None
    value = np.asarray(payload[name])
    if value.size != 1:
        return False
    return str(value.reshape(-1)[0]) == expected


def _cache_hash_states(
    cfg: DataConfig, payload: Mapping[str, np.ndarray]
) -> tuple[bool | None, bool | None]:
    vocab = int(np.asarray(payload["vocab"]).reshape(-1)[0])
    ids = np.asarray(payload["ids"]).reshape(cfg.gen_sequences, cfg.gen_len)
    masks = (
        {
            "constrained": np.asarray(payload["mask"]).reshape(
                cfg.gen_sequences, cfg.gen_len
            )
        }
        if cfg.kind == "contextual"
        else {}
    )
    expected_content = dataset_content_hash(
        ids,
        masks,
        tokenizer=IdentityTokenizer(vocab),
    )
    return (
        _stored_hash_state(payload, "_config_hash", data_config_hash(cfg)),
        _stored_hash_state(payload, "_content_hash", expected_content),
    )


def _load_cache(
    cfg: DataConfig, legacy_key: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    current, legacy = _cache_paths(cfg, legacy_key)
    for path, identity in ((current, "config_sha256"), (legacy, "legacy_key")):
        if not path.exists():
            continue
        if identity == "legacy_key" and not _legacy_cache_identity_is_exact(cfg):
            continue
        with np.load(path, allow_pickle=False) as loaded:
            payload = {name: np.array(loaded[name], copy=True) for name in loaded.files}
        if not _cache_payload_is_valid(cfg, payload):
            if identity == "config_sha256":
                raise ValueError(f"Invalid dataset cache payload at {path}.")
            continue
        config_hash_state, content_hash_state = _cache_hash_states(cfg, payload)
        if config_hash_state is False or content_hash_state is False:
            raise ValueError(f"Dataset cache identity mismatch at {path}.")
        return payload, {
            "source": "cache",
            "cache_path": str(path),
            "cache_identity": identity,
            "legacy_identity_unverified": identity == "legacy_key",
            "config_identity_verified": config_hash_state is True,
            "content_identity_verified": content_hash_state is True,
        }
    return None


def _write_cache(
    cfg: DataConfig, legacy_key: str, **payload: np.ndarray | int
) -> tuple[Path, dict[str, Any]]:
    current, _ = _cache_paths(cfg, legacy_key)
    enriched = dict(payload)
    if {"ids", "vocab"} <= set(enriched):
        vocab = int(np.asarray(enriched["vocab"]).reshape(-1)[0])
        ids = np.asarray(enriched["ids"]).reshape(
            cfg.gen_sequences, cfg.gen_len
        )
        masks = (
            {
                "constrained": np.asarray(enriched["mask"]).reshape(
                    cfg.gen_sequences, cfg.gen_len
                )
            }
            if cfg.kind == "contextual" and "mask" in enriched
            else {}
        )
        enriched["_config_hash"] = np.asarray(data_config_hash(cfg))
        enriched["_content_hash"] = np.asarray(
            dataset_content_hash(
                ids,
                masks,
                tokenizer=IdentityTokenizer(vocab),
            )
        )
    np.savez(current, **enriched)
    return current, {
        "source": "generated",
        "cache_path": str(current),
        "cache_identity": "config_sha256",
        "legacy_identity_unverified": False,
        "config_identity_verified": True,
        "content_identity_verified": True,
    }


def _synthetic_bundle(
    cfg: DataConfig,
    ids: np.ndarray,
    vocab: int,
    *,
    generator: str,
    provenance: Mapping[str, Any],
    masks: Mapping[str, np.ndarray] | None = None,
) -> DatasetBundle:
    shape = (cfg.gen_sequences, cfg.gen_len)
    ids = np.asarray(ids, dtype=np.int32).reshape(shape)
    reshaped_masks = {
        name: np.asarray(mask).reshape(shape) for name, mask in (masks or {}).items()
    }
    return _bundle(
        cfg,
        ids,
        IdentityTokenizer(vocab),
        masks=reshaped_masks,
        provenance={**dict(provenance), "generator": generator},
    )


def load_dataset_bundle(cfg: DataConfig) -> DatasetBundle:
    """Load a canonical bundle while retaining independent trajectories."""
    if cfg.kind == "text":
        path = Path(cfg.corpus_path)
        text = load_corpus(path)
        tokenizer = CharTokenizer(text)
        return _bundle(
            cfg,
            tokenizer.encode(text),
            tokenizer,
            provenance={
                "source": "file" if path.exists() else "synthetic_fallback",
                "corpus_path": str(path),
            },
        )

    if cfg.kind == "seq_cancellation":
        from .seq_cancellation import seq_cancellation

        legacy_key = (
            f"seqcancel_v{cfg.ctx_observables}_w{cfg.ctx_context_size}"
            f"_d{cfg.seq_cancel_density:.3f}_s{cfg.gen_sequences}"
            f"_l{cfg.gen_len}_seed{cfg.gen_seed}"
        )
        cached = _load_cache(cfg, legacy_key)
        if cached is not None:
            payload, provenance = cached
            return _synthetic_bundle(
                cfg, payload["ids"], int(payload["vocab"].item()),
                generator="seq_cancellation", provenance=provenance,
            )
        ids, vocab = seq_cancellation(
            n_sequences=cfg.gen_sequences,
            seq_len=cfg.gen_len,
            vocab_size=cfg.ctx_observables,
            parity_window=cfg.ctx_context_size,
            density=cfg.seq_cancel_density,
            seed=cfg.gen_seed,
        )
        ids = np.asarray(ids, dtype=np.int32)
        _, provenance = _write_cache(cfg, legacy_key, ids=ids, vocab=vocab)
        return _synthetic_bundle(
            cfg, ids, vocab, generator="seq_cancellation", provenance=provenance
        )

    if cfg.kind == "interference":
        from .interference_task import interference_sequences

        legacy_key = (
            f"interference_v{cfg.ctx_observables}_s{cfg.gen_sequences}"
            f"_l{cfg.gen_len}_seed{cfg.gen_seed}"
        )
        cached = _load_cache(cfg, legacy_key)
        if cached is not None:
            payload, provenance = cached
            return _synthetic_bundle(
                cfg, payload["ids"], int(payload["vocab"].item()),
                generator="interference_sequences", provenance=provenance,
            )
        ids, vocab = interference_sequences(
            n_sequences=cfg.gen_sequences,
            seq_len=cfg.gen_len,
            vocab_size=cfg.ctx_observables,
            seed=cfg.gen_seed,
        )
        ids = np.asarray(ids, dtype=np.int32)
        _, provenance = _write_cache(cfg, legacy_key, ids=ids, vocab=vocab)
        return _synthetic_bundle(
            cfg, ids, vocab,
            generator="interference_sequences", provenance=provenance,
        )

    if cfg.kind == "contextual":
        from .contextual import contextual_parity_sequences

        legacy_key = (
            f"contextual_o{cfg.ctx_observables}_c{cfg.ctx_context_size}"
            f"_live{cfg.ctx_n_live}_s{cfg.gen_sequences}_l{cfg.gen_len}"
            f"_seed{cfg.gen_seed}"
        )
        cached = _load_cache(cfg, legacy_key)
        if cached is not None:
            payload, provenance = cached
            return _synthetic_bundle(
                cfg,
                payload["ids"],
                int(payload["vocab"].item()),
                masks={"constrained": payload["mask"]},
                generator="contextual_parity_sequences",
                provenance=provenance,
            )
        ids, vocab, mask = contextual_parity_sequences(
            n_sequences=cfg.gen_sequences,
            seq_len=cfg.gen_len,
            n_observables=cfg.ctx_observables,
            context_size=cfg.ctx_context_size,
            n_live=cfg.ctx_n_live,
            seed=cfg.gen_seed,
        )
        ids = np.asarray(ids, dtype=np.int32)
        mask = np.asarray(mask, dtype=np.int32)
        _, provenance = _write_cache(
            cfg, legacy_key, ids=ids, vocab=vocab, mask=mask
        )
        return _synthetic_bundle(
            cfg,
            ids,
            vocab,
            masks={"constrained": mask},
            generator="contextual_parity_sequences",
            provenance=provenance,
        )

    if cfg.kind in ("monitored_ising", "markov_control"):
        legacy_key = (
            f"{cfg.kind}{'_boundary_v2' if cfg.kind == 'markov_control' else ''}"
            f"_q{cfg.gen_qubits}_m{cfg.gen_measured}"
            f"_s{cfg.gen_sequences}_l{cfg.gen_len}"
            f"_zz{cfg.gen_theta_zz:.4f}_x{cfg.gen_theta_x:.4f}"
            f"_spt{cfg.gen_steps_per_token}_seed{cfg.gen_seed}"
            f"_k{cfg.markov_order}"
        )
        cached = _load_cache(cfg, legacy_key)
        if cached is not None:
            payload, provenance = cached
            return _synthetic_bundle(
                cfg,
                payload["ids"],
                int(payload["vocab"].item()),
                generator=(
                    "markov_control_sequences"
                    if cfg.kind == "markov_control"
                    else "monitored_ising_sequences"
                ),
                provenance=provenance,
            )

        ids, vocab = monitored_ising_sequences(
            n_qubits=cfg.gen_qubits,
            n_measured=cfg.gen_measured,
            n_sequences=cfg.gen_sequences,
            seq_len=cfg.gen_len,
            theta_zz=cfg.gen_theta_zz,
            theta_x=cfg.gen_theta_x,
            steps_per_token=cfg.gen_steps_per_token,
            seed=cfg.gen_seed,
        )
        if cfg.kind == "markov_control":
            ids = ids.reshape(cfg.gen_sequences, cfg.gen_len)
            ids = markov_control_sequences(
                ids, vocab, order=cfg.markov_order, seed=cfg.gen_seed + 1
            )
        ids = np.asarray(ids, dtype=np.int32)
        _, provenance = _write_cache(cfg, legacy_key, ids=ids, vocab=vocab)
        return _synthetic_bundle(
            cfg,
            ids,
            vocab,
            generator=(
                "markov_control_sequences"
                if cfg.kind == "markov_control"
                else "monitored_ising_sequences"
            ),
            provenance=provenance,
        )

    raise ValueError(
        f"Unknown data kind '{cfg.kind}'. Options: {choices_text(DATASET_KINDS)}"
    )


def load_dataset(cfg: DataConfig):
    """Compatibility API returning a flat token stream and tokenizer."""
    bundle = load_dataset_bundle(cfg)
    return bundle.ids.reshape(-1), bundle.tokenizer
