"""Feature shards: the handoff from curation to Milestone 4's index build.

Embeddings are written as ``.npz`` shards of ``config.embedding_shard_size``
assets under ``indexes/<dataset_version>/<encoder_key>/``:

```
indexes/2026.09.06-colab1/
  mobileclip2_s2/
    index.json          # model card, dims, shard list, counts
    shard-000000.npz    # asset_ids (str) + features (float32, L2-normalised)
    shard-000001.npz
  dinov2_vits14/
    ...
```

Three properties matter on a free Colab tier and shaped this layout:

* **Resumable.** ``existing_ids()`` reads the shard index, so a restarted
  runtime embeds only what is missing instead of re-running the whole gallery.
* **Sharded.** A disconnect loses at most one shard, and the index is rewritten
  atomically after each shard lands, so a partial file is never referenced.
* **Self-describing.** ``index.json`` carries the full model card (including the
  licence string) next to the features it produced, which is what makes an
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


class EmbeddingStore:
    """Append-only feature shards plus their index."""

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

    def existing_ids(self) -> set[str]:
        """Asset ids that already have a stored feature vector."""
        ids: set[str] = set()
        for shard in self.shards():
            path = self.root / shard.path
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as data:
                stored = data["asset_ids"] if "asset_ids" in data else np.array([])
            ids.update(str(value) for value in stored.tolist())
        return ids

    def append(self, asset_ids: list[str], features: np.ndarray) -> ShardRef:
        """Write one shard and extend the index. Both writes are atomic."""
        if len(asset_ids) != features.shape[0]:
            msg = f"{len(asset_ids)} ids but features shaped {features.shape}"
            raise EmbeddingStoreError(msg)
        if features.ndim != 2:
            msg = f"features must be 2-D, got shape {features.shape}"
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


def chunked(items: list[str], size: int) -> list[list[str]]:
    """Split ids into shard-sized batches."""
    if size <= 0:
        msg = "shard size must be positive"
        raise EmbeddingStoreError(msg)
    return [items[start : start + size] for start in range(0, len(items), size)]
