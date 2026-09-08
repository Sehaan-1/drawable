"""Feature shards: the handoff from curation to Milestone 4's index build.

Embeddings are written as ``.npz`` shards of ``config.embedding_shard_size``
assets under ``indexes/<dataset_version>/<encoder_key>/``:

```
indexes/2026.09.06-colab1/
  mobileclip2_s2/
    index.json          # model card, dims, shard list, counts, artifact entries
    shard-000000.npz    # asset_ids (str) + features (float32, L2-normalised)
    shard-000001.npz
  dinov2_vits14/
    ...
```

Three properties matter on a free Colab tier and shaped this layout:

* **Resumable.** ``existing_ids()`` reads the shard index, so a restarted
  runtime embeds only what is missing instead of re-running the whole gallery.
  Entries are stamped with the artifact they were computed from, and a stamp
  that no longer matches (the asset was re-extracted) does **not** count as
  done — stale vectors are re-embedded, never reused.
* **Sharded.** A disconnect loses at most one shard, and the index is rewritten
  atomically after each shard lands, so a partial file is never referenced.
* **Self-describing.** ``index.json`` carries the full model card (including
  the licence string) next to the features it produced, which is what makes an
  index buildable later without guessing which checkpoint generated it.

Features come from the *line art*, not the original: that is the image the API
serves and the one a query sketch is compared against.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from linescout_ml.colab.models import MODEL_CARDS, ModelCard
from linescout_ml.embeddings import ArtifactStamp, EmbeddingEntry, EmbeddingIndex

INDEX_FILENAME = "index.json"
SHARD_TEMPLATE = "shard-{index:06d}.npz"


class EmbeddingStoreError(RuntimeError):
    """A shard or index could not be read, or its shape disagrees with the index."""


@dataclass(frozen=True)
class ShardRef:
    """One shard as recorded in ``index.json``."""

    path: str
    count: int


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _as_int(value: object) -> int:
    """Index files are JSON; a hand-edited ``count`` must not crash a resume."""
    return int(value) if isinstance(value, int) else 0


def _entries_from_index(index: dict[str, object]) -> dict[str, EmbeddingEntry]:
    """Per-asset artifact bindings recorded by :meth:`EmbeddingStore.append`."""
    raw = index.get("entries", {})
    if not isinstance(raw, dict):
        return {}
    entries: dict[str, EmbeddingEntry] = {}
    for asset_id, value in raw.items():
        if not isinstance(value, dict):
            continue
        try:
            entries[asset_id] = EmbeddingEntry.model_validate(value)
        except ValueError:
            continue  # a malformed entry is treated as absent, never as valid
    return entries


class EmbeddingStore:
    """Append-only feature shards plus their index.

    Every appended entry is stamped with the artifact it was computed from
    (``processing_revision`` + ``line_art_checksum``), implementing the frozen
    vector-payload rule from :mod:`linescout_ml.embeddings`: a vector whose
    artifact binding no longer matches the asset is stale, and stale means
    re-embed — never serve.
    """

    def __init__(self, root: Path, card: ModelCard) -> None:
        self.root = root
        self.card = card
        self.observed_dim = card.dim

    @classmethod
    def for_key(cls, root: Path, key: str) -> EmbeddingStore:
        try:
            card = MODEL_CARDS[key]
        except KeyError:
            msg = f"unknown encoder {key!r}; available: {sorted(MODEL_CARDS)}"
            raise EmbeddingStoreError(msg) from None
        return cls(root / key, card)

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    def _read_index(self) -> dict[str, object]:
        if not self.index_path.is_file():
            return {}
        try:
            loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            msg = f"unreadable embedding index {self.index_path}: {error}"
            raise EmbeddingStoreError(msg) from error
        return loaded if isinstance(loaded, dict) else {}

    def shards(self) -> list[ShardRef]:
        raw = self._read_index().get("shards", [])
        if not isinstance(raw, list):
            return []
        refs: list[ShardRef] = []
        for entry in raw:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            count = entry.get("count")
            refs.append(
                ShardRef(
                    path=str(entry["path"]),
                    count=int(count) if isinstance(count, int) else 0,
                )
            )
        return refs

    def load_entries(self) -> dict[str, EmbeddingEntry]:
        """The per-asset artifact bindings recorded so far."""
        return _entries_from_index(self._read_index())

    def existing_ids(self, artifacts: dict[str, ArtifactStamp] | None = None) -> set[str]:
        """Asset ids that already have a *usable* stored feature vector.

        Without ``artifacts``, every recorded id counts (legacy behaviour).
        With it, an id only counts when its entry's artifact stamp matches the
        current one — stale entries are re-embedded, never reused.
        """
        if artifacts is None:
            ids: set[str] = set()
            for shard in self.shards():
                path = self.root / shard.path
                if not path.is_file():
                    continue
                with np.load(path, allow_pickle=False) as data:
                    stored = data["asset_ids"] if "asset_ids" in data else np.array([])
                ids.update(str(value) for value in stored.tolist())
            return ids
        entries = self.load_entries()
        return {
            asset_id
            for asset_id, stamp in artifacts.items()
            if asset_id in entries
            and entries[asset_id].artifact.processing_revision == stamp.processing_revision
            and entries[asset_id].artifact.line_art_checksum == stamp.line_art_checksum
        }

    def append(
        self,
        asset_ids: list[str],
        features: np.ndarray,
        stamps: list[ArtifactStamp] | None = None,
    ) -> ShardRef:
        """Write one shard and extend the index. Both writes are atomic."""
        if len(asset_ids) != features.shape[0]:
            msg = f"{len(asset_ids)} ids but features shaped {features.shape}"
            raise EmbeddingStoreError(msg)
        if features.ndim != 2:
            msg = f"features must be 2-D, got shape {features.shape}"
            raise EmbeddingStoreError(msg)
        if stamps is not None and len(stamps) != len(asset_ids):
            msg = f"{len(asset_ids)} ids but {len(stamps)} artifact stamps"
            raise EmbeddingStoreError(msg)

        dim = int(features.shape[1])
        if self.observed_dim and dim != self.observed_dim:
            msg = f"{self.card.key} shard dim {dim} disagrees with existing {self.observed_dim}"
            raise EmbeddingStoreError(msg)
        self.observed_dim = dim

        self.root.mkdir(parents=True, exist_ok=True)
        existing = self.shards()
        name = SHARD_TEMPLATE.format(index=len(existing))
        temporary = self.root / (name + ".tmp.npz")
        np.savez(
            temporary,
            asset_ids=np.array(asset_ids, dtype=np.str_),
            features=np.ascontiguousarray(features, dtype=np.float32),
        )
        os.replace(temporary, self.root / name)

        index = self._read_index()
        shards = [{"path": shard.path, "count": shard.count} for shard in existing]
        shards.append({"path": name, "count": len(asset_ids)})
        entries = _entries_from_index(index)
        if stamps is not None:
            for asset_id, stamp in zip(asset_ids, stamps, strict=True):
                entries[asset_id] = EmbeddingEntry(artifact=stamp, shard=name)
        payload = {
            "schema_version": 1,
            "spec": {
                "key": self.card.key,
                "family": self.card.family,
                "name": self.card.name,
                "pretrained": self.card.pretrained,
                "dim": dim,
                "image_size": self.card.image_size,
                "license": self.card.license,
                "upstream": self.card.upstream,
                "citation": self.card.citation,
            },
            "dtype": "float32",
            "normalised": True,
            "count": _as_int(index.get("count")) + len(asset_ids),
            "created_at": index.get("created_at") or _now(),
            "updated_at": _now(),
            "shards": shards,
            "entries": {
                asset_id: entry.model_dump() for asset_id, entry in sorted(entries.items())
            },
        }
        _atomic_write_text(self.index_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return ShardRef(path=name, count=len(asset_ids))

    def load_all(self) -> tuple[list[str], np.ndarray]:
        """Every stored feature, in shard order — the input to an index build."""
        ids: list[str] = []
        blocks: list[np.ndarray] = []
        for shard in self.shards():
            path = self.root / shard.path
            if not path.is_file():
                msg = f"index references a missing shard: {path}"
                raise EmbeddingStoreError(msg)
            with np.load(path, allow_pickle=False) as data:
                ids.extend(str(value) for value in data["asset_ids"].tolist())
                blocks.append(data["features"].astype(np.float32))
        if not blocks:
            return [], np.zeros((0, self.observed_dim or 0), dtype=np.float32)
        features = np.concatenate(blocks, axis=0)
        if features.shape[0] != len(ids):
            msg = f"shards hold {features.shape[0]} rows for {len(ids)} ids"
            raise EmbeddingStoreError(msg)
        return ids, features

    def embedding_index(self) -> EmbeddingIndex:
        """The shard index as the contract's :class:`EmbeddingIndex`."""
        return EmbeddingIndex(embedder=self.card.key, entries=self.load_entries())


def chunked(items: list[str], size: int) -> list[list[str]]:
    """Split ids into shard-sized batches."""
    if size <= 0:
        msg = "shard size must be positive"
        raise EmbeddingStoreError(msg)
    return [items[start : start + size] for start in range(0, len(items), size)]
