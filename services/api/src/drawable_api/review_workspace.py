from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from drawable_research_contracts.readiness import SourceSelectionProfileV1

from . import intake
from .catalog import catalog_manifest
from .settings import Settings

SCHEMA_VERSION = 1
MAX_TERMS_BYTES = 10 * 1024 * 1024
MAX_CANDIDATE_BATCH = 32
PERMISSIONS = (
    "local_display",
    "training",
    "evaluation",
    "derive",
    "trace",
    "export",
    "public_demo",
    "redistribute",
    "cloud_storage",
    "cloud_processing",
)
DEFAULT_WORK_SELECTION = ("hero", "lma_test", "pirate")
DEFAULT_SOURCE_SELECTION = ("anita-dataset",)
DEFAULT_MODEL_SELECTION = ("mobileclip2-s2-dfndr2b", "dinov2-small")
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,80}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReviewWorkspaceError(ValueError):
    pass


def _read_document(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReviewWorkspaceError("A required local review configuration is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewWorkspaceError("A local review configuration is not valid JSON") from error
    if not isinstance(value, dict):
        raise ReviewWorkspaceError("A local review configuration must be a JSON object")
    return value


def _write_private_json(root: Path, path: Path, value: Any) -> None:
    resolved_root = root.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or resolved_root not in path.resolve(strict=False).parents:
        raise ReviewWorkspaceError("Private review files must stay beneath the selected data root")
    os.chmod(path.parent, 0o700)
    if path.is_symlink():
        raise ReviewWorkspaceError("Private review files may not be symbolic links")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _permissions(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != set(PERMISSIONS):
        raise ReviewWorkspaceError("Choose a decision for every listed use")
    if any(type(value.get(key)) is not bool for key in PERMISSIONS):
        raise ReviewWorkspaceError("Use decisions must be on or off")
    if value["cloud_processing"] and not value["cloud_storage"]:
        raise ReviewWorkspaceError("Cloud processing requires separate storage permission")
    return {key: value[key] for key in PERMISSIONS}


def _lock_for(root: Path) -> threading.RLock:
    key = str(root.resolve())
    with _LOCK_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


_LOCK_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


class ReviewWorkspace:
    """Private owner decisions and effective registries for local reference review."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.data_root.resolve()
        self.directory = self.root / "reports" / "private" / "review-workspace"
        self.decisions_path = self.directory / "owner-decisions-v1.json"
        self.sources_path = self.directory / "sources-effective.json"
        self.works_path = self.directory / "anita-work-terms-effective.json"
        self.models_path = self.directory / "models-effective.json"
        self.profile_path = self.directory / "selected-source-profile.json"
        self._lock = _lock_for(self.root)
        self._source_defaults = _read_document(settings.source_registry_path)
        self._work_defaults = _read_document(settings.anita_work_terms_path)
        self._model_defaults = _read_document(settings.model_registry_path)
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            for directory in (self.root / "reports", self.root / "reports" / "private", self.directory):
                if directory.is_symlink():
                    raise ReviewWorkspaceError("Private review directories may not be symbolic links")
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(directory, 0o700)
            if not self.decisions_path.exists():
                _write_private_json(
                    self.root,
                    self.decisions_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "revision": 0,
                        "selection": {
                            "sources": list(DEFAULT_SOURCE_SELECTION),
                            "works": list(DEFAULT_WORK_SELECTION),
                            "models": list(DEFAULT_MODEL_SELECTION),
                        },
                        "source_decisions": {},
                        "work_decisions": {},
                        "model_decisions": {},
                        "captured_terms": {},
                        "events": [],
                    },
                )
            state = self._state()
            preparation = state.get("preparation", {})
            if isinstance(preparation, dict) and preparation.get("status") in {"queued", "running"}:
                preparation.update(
                    status="blocked",
                    phase="interrupted",
                    message="Preparation stopped when the local app closed. Its partial download is safe to resume.",
                    resumable=True,
                )
                state["revision"] += 1
                _write_private_json(self.root, self.decisions_path, state)
            if not self.profile_path.exists():
                self._save_profile(state)
            self._materialize(state)

    def _state(self) -> dict[str, Any]:
        state = _read_document(self.decisions_path)
        if state.get("schema_version") != SCHEMA_VERSION or not isinstance(state.get("revision"), int):
            raise ReviewWorkspaceError("The private review workspace has an unsupported schema")
        for name in ("selection", "source_decisions", "work_decisions", "model_decisions"):
            if not isinstance(state.get(name), dict):
                raise ReviewWorkspaceError("The private review workspace is malformed")
        if not isinstance(state.get("events"), list):
            raise ReviewWorkspaceError("The private review history is malformed")
        if not isinstance(state.get("captured_terms", {}), dict):
            raise ReviewWorkspaceError("The private terms evidence index is malformed")
        state.setdefault("captured_terms", {})
        return state

    def _save_profile(self, state: dict[str, Any]) -> None:
        selection = state["selection"]
        registry_rows = self._source_defaults.get("sources", [])
        work_rows = self._work_defaults.get("works", [])
        sources = {row.get("id"): row for row in registry_rows if isinstance(row, dict)}
        works = {row.get("folder_id"): row for row in work_rows if isinstance(row, dict)}
        selected_sources = selection.get("sources", [])
        selected_works = selection.get("works", [])
        if (
            not isinstance(selected_sources, list)
            or not selected_sources
            or any(item not in sources for item in selected_sources)
            or not isinstance(selected_works, list)
            or not selected_works
            or any(item not in works for item in selected_works)
        ):
            raise ReviewWorkspaceError("Select at least one known source and one known work")
        profile = SourceSelectionProfileV1.model_validate(
            {
                "schema_version": 1,
                "profile_id": "local-reference-review-v1",
                "sources": [
                    {
                        "source_id": source_id,
                        "acquisition_method": "official_archive",
                        "work_ids": selected_works if source_id == "anita-dataset" else [],
                        "operations": ["local_display"],
                        "acquisition_evidence": [],
                    }
                    for source_id in selected_sources
                ],
            }
        )
        _write_private_json(self.root, self.profile_path, profile.model_dump(mode="json"))

    def _materialize(self, state: dict[str, Any]) -> None:
        sources = json.loads(json.dumps(self._source_defaults))
        works = json.loads(json.dumps(self._work_defaults))
        models = json.loads(json.dumps(self._model_defaults))
        for row in sources.get("sources", []):
            captured = state["captured_terms"].get(f"source:{row.get('id')}")
            if isinstance(captured, dict) and isinstance(captured.get("sha256"), str):
                row["terms_snapshot_sha256"] = captured["sha256"]
            decision = state["source_decisions"].get(row.get("id"))
            if decision:
                stale = decision["terms_snapshot_sha256"] != row.get("terms_snapshot_sha256")
                row.update(
                    rights_status="review_required" if stale else decision["decision"],
                    terms_review={
                        "status": "review_required" if stale else decision["decision"],
                        "reviewer_id": decision["reviewer_id"],
                        "reviewed_at": decision["reviewed_at"],
                        "reviewed_sha256": decision["terms_snapshot_sha256"],
                    },
                    use_permissions=dict.fromkeys(PERMISSIONS, False) if stale else decision["permissions"],
                )
        for row in works.get("works", []):
            decision = state["work_decisions"].get(row.get("folder_id"))
            if decision:
                source_capture = state["captured_terms"].get("source:anita-dataset")
                source_hash = source_capture.get("sha256") if isinstance(source_capture, dict) else None
                stale = decision["terms_snapshot_sha256"] != source_hash
                row.update(
                    rights_status="review_required" if stale else decision["decision"],
                    reviewer_decision="review_required" if stale else decision["decision"],
                    reviewer_id=decision["reviewer_id"],
                    reviewed_at=decision["reviewed_at"],
                    reviewed_sha256=decision["terms_snapshot_sha256"],
                    terms_snapshot_sha256=source_hash,
                    attribution_text=decision["attribution_text"],
                    use_permissions=dict.fromkeys(PERMISSIONS, False) if stale else decision["permissions"],
                )
        for row in models.get("models", []):
            captured = state["captured_terms"].get(f"model:{row.get('id')}")
            if isinstance(captured, dict) and isinstance(captured.get("sha256"), str):
                row["terms_review"]["terms_snapshot_sha256"] = captured["sha256"]
            decision = state["model_decisions"].get(row.get("id"))
            if decision:
                stale = decision["terms_snapshot_sha256"] != row["terms_review"].get("terms_snapshot_sha256")
                row["terms_review"] = {
                    "status": "pending" if stale else decision["decision"],
                    "reviewer_id": decision["reviewer_id"],
                    "reviewed_at": decision["reviewed_at"],
                    "terms_snapshot_sha256": captured["sha256"] if stale else decision["terms_snapshot_sha256"],
                    "review_note": decision["note"],
                }
                # This review UI cannot grant training permission.
                row["training_authorized"] = False
                row["cloud_training_authorized"] = False
        _write_private_json(self.root, self.sources_path, sources)
        _write_private_json(self.root, self.works_path, works)
        _write_private_json(self.root, self.models_path, models)

    def refresh(self) -> None:
        with self._lock:
            state = self._state()
            self._materialize(state)

    def profile(self) -> dict[str, Any]:
        return _read_document(self.profile_path)

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state()
            self._materialize(state)
            selected = state["selection"]
            sources = {row["id"]: row for row in self.sources()["sources"]}
            works = {row["folder_id"]: row for row in self.works()["works"]}
            models = {row["id"]: row for row in self.models()["models"]}
            archive_audit = intake.audit_sources(self.root, registry_path=self.sources_path, verify_archives=False)
            archives = {item["source_id"]: item for item in archive_audit.get("sources", [])}
            candidates = self._candidate_counts()
            return {
                "schema_version": 1,
                "revision": state["revision"],
                "selected_sources": [self._source_status(sources[item], archives.get(item), candidates) for item in selected["sources"]],
                "selected_works": [self._work_status(works[item]) for item in selected["works"]],
                "available_works": [
                    {**self._work_status(row), "selected": row["folder_id"] in selected["works"]}
                    for row in self.works()["works"]
                ],
                "selected_models": [self._model_status(models[item]) for item in selected["models"]],
                "reviewer_id": state.get("reviewer_id", ""),
                "preparation": state.get("preparation", {"status": "not_started"}),
                "curation": self._curation_progress(),
                "required_handoff": "Review source and work permissions, then prepare images for local review.",
                "private_storage": True,
                "training_authorized": False,
            }

    def sources(self) -> dict[str, Any]:
        return _read_document(self.sources_path)

    def works(self) -> dict[str, Any]:
        return _read_document(self.works_path)

    def models(self) -> dict[str, Any]:
        return _read_document(self.models_path)

    def _source_status(self, row: dict[str, Any], archive: Any, candidates: dict[str, int]) -> dict[str, Any]:
        return {
            "id": row["id"], "title": row["title"], "authority_url": row["authority_url"],
            "terms_url": row.get("terms_url") or row["authority_url"],
            "decision": row.get("rights_status", "review_required"),
            "terms_snapshot_sha256": row.get("terms_snapshot_sha256"),
            "permissions": row["use_permissions"], "archive": archive,
            "candidate_count": candidates.get(row["id"], 0),
        }

    def _work_status(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["folder_id"], "title": row["work_title"], "artist": row["artist"],
            "license": row["license"], "decision": row.get("rights_status", "review_required"),
            "permissions": row["use_permissions"], "attribution_text": row.get("attribution_text"),
        }

    def _model_status(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"], "title": row["variant"], "terms_url": row["terms_source_url"],
            "decision": row.get("terms_review", {}).get("status", "pending"),
            "terms_snapshot_sha256": row.get("terms_review", {}).get("terms_snapshot_sha256"),
            "weights_state": row.get("state", "not_acquired"),
            "training_authorized": False,
        }

    def _candidate_counts(self) -> dict[str, int]:
        database = self.settings.metadata_db
        if not database.is_file() or database.is_symlink():
            return {}
        from .database import connect

        db = connect(database)
        try:
            return {
                row["source_id"]: row["count"]
                for row in db.execute("SELECT source_id, COUNT(*) AS count FROM assets GROUP BY source_id")
            }
        finally:
            db.close()

    def _curation_progress(self) -> dict[str, int]:
        try:
            from .curation import curation_progress
            return curation_progress(self.settings)
        except (OSError, ValueError, RuntimeError):
            return {"total": 0, "pending": 0, "candidate": 0, "rejected": 0, "reviewed": 0}

    def _snapshot(self, kind: str, identifier: str) -> tuple[str, str, str]:
        """Return terms path, the exact expected URL, and current snapshot digest."""
        if kind == "source":
            source = next((row for row in self._source_defaults["sources"] if row["id"] == identifier), None)
            if source is None:
                raise ReviewWorkspaceError("Unknown source")
            if identifier not in self._state()["selection"]["sources"]:
                raise ReviewWorkspaceError("Select this source before reviewing its terms")
            expected_url = source.get("terms_url") or source["authority_url"]
            digest = self.sources()["sources"]
            row = next(item for item in digest if item["id"] == identifier)
            return "source", expected_url, str(row.get("terms_snapshot_sha256") or "")
        if kind == "work":
            selected_source = self._source_defaults["sources"][0]["id"]
            source = self._snapshot("source", selected_source)
            return "source", source[1], source[2]
        if kind == "model":
            row = next((item for item in self._model_defaults["models"] if item["id"] == identifier), None)
            if row is None:
                raise ReviewWorkspaceError("Unknown model")
            if identifier not in self._state()["selection"]["models"]:
                raise ReviewWorkspaceError("Select this model before reviewing its terms")
            effective = next(item for item in self.models()["models"] if item["id"] == identifier)
            return "model", row["terms_source_url"], str(effective.get("terms_review", {}).get("terms_snapshot_sha256") or "")
        raise ReviewWorkspaceError("Terms type must be source, work, or model")

    def capture_terms(self, kind: str, identifier: str) -> dict[str, Any]:
        if kind not in {"source", "model"}:
            raise ReviewWorkspaceError("Capture terms for the parent source; work records share its terms evidence")
        with self._lock:
            if kind == "source":
                receipt = intake.capture_terms_snapshot(self.root, identifier, registry_path=self.sources_path)
                snapshot_path = intake.ensure_inside(self.root, receipt["snapshot_path"])
                receipt_path = intake.terms_snapshot_receipt_path(
                    self.root, identifier, receipt["sha256"]
                )
                self._secure_private_file(snapshot_path)
                self._secure_private_file(receipt_path)
                result = {
                    **receipt,
                    "kind": "source",
                    "terms_snapshot_sha256": receipt["sha256"],
                    "owner_decision_required": True,
                }
            else:
                result = self._capture_model_terms(identifier)
            state = self._state()
            state["captured_terms"][f"{kind}:{identifier}"] = {
                "sha256": result["terms_snapshot_sha256"],
                "captured_at": result.get("retrieved_at") or result.get("captured_at"),
                "snapshot_path": result.get("snapshot_path"),
            }
            state["revision"] += 1
            _write_private_json(self.root, self.decisions_path, state)
            self._materialize(state)
            return result

    def _secure_private_file(self, path: Path) -> None:
        resolved_root = self.root.resolve()
        resolved = path.resolve(strict=True)
        if resolved_root not in resolved.parents or path.is_symlink() or not path.is_file():
            raise ReviewWorkspaceError("Private evidence must be a regular file inside the data root")
        cursor = path.parent
        while cursor != self.root:
            if cursor.is_symlink():
                raise ReviewWorkspaceError("Private evidence paths may not traverse symbolic links")
            if self.root not in cursor.parents:
                raise ReviewWorkspaceError("Private evidence must stay inside the data root")
            cursor = cursor.parent
        os.chmod(path.parent, 0o700)
        os.chmod(path, 0o600)

    def terms_document(self, kind: str, identifier: str) -> tuple[bytes, str]:
        """Read the exact captured terms bytes for a sandboxed local review view."""
        if kind == "source":
            source = next((row for row in self.sources()["sources"] if row["id"] == identifier), None)
            if source is None:
                raise ReviewWorkspaceError("Unknown source")
            digest = str(source.get("terms_snapshot_sha256") or "")
            if not DIGEST_PATTERN.fullmatch(digest):
                raise ReviewWorkspaceError("Capture the official terms page before opening its saved copy")
            receipt_path = intake.terms_snapshot_receipt_path(self.root, identifier, digest)
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ReviewWorkspaceError("The captured source terms receipt is missing")
            receipt = _read_document(receipt_path)
            if receipt.get("source_id") != identifier or receipt.get("sha256") != digest:
                raise ReviewWorkspaceError("The source terms receipt does not match the current capture")
            snapshot_path = intake.ensure_inside(self.root, str(receipt.get("snapshot_path", "")))
            self._secure_private_file(snapshot_path)
            if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != digest:
                raise ReviewWorkspaceError("The captured source terms bytes failed their hash check")
            content_type = str(receipt.get("content_type") or "text/plain")
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain", "application/json"}:
                content_type = "text/plain"
            return snapshot_path.read_bytes(), content_type

        if kind != "model":
            raise ReviewWorkspaceError("Terms type must be source or model")
        model = next((row for row in self.models()["models"] if row["id"] == identifier), None)
        if model is None:
            raise ReviewWorkspaceError("Unknown model")
        digest = str(model.get("terms_review", {}).get("terms_snapshot_sha256") or "")
        if not DIGEST_PATTERN.fullmatch(digest):
            raise ReviewWorkspaceError("Capture the pinned model page before opening its saved copy")
        receipt_path = self.root / "manifests" / "model-terms" / identifier / f"{digest}.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ReviewWorkspaceError("The captured model terms receipt is missing")
        self._secure_private_file(receipt_path)
        if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != digest:
            raise ReviewWorkspaceError("The captured model terms receipt failed its hash check")
        receipt = _read_document(receipt_path)
        if (
            receipt.get("artifact_type") != "drawable-model-terms-snapshot"
            or receipt.get("model_id") != identifier
            or receipt.get("purpose") != "inference"
        ):
            raise ReviewWorkspaceError("The captured model terms receipt has the wrong identity")
        try:
            content = base64.b64decode(str(receipt["content_base64"]), validate=True)
        except (KeyError, ValueError) as error:
            raise ReviewWorkspaceError("The captured model terms page is malformed") from error
        if hashlib.sha256(content).hexdigest() != receipt.get("content_sha256"):
            raise ReviewWorkspaceError("The captured model terms page failed its hash check")
        content_type = str(receipt.get("content_type") or "text/plain")
        if content_type not in {"text/html", "application/xhtml+xml", "text/plain", "application/json"}:
            content_type = "text/plain"
        return content, content_type

    def _capture_model_terms(self, model_id: str) -> dict[str, Any]:
        defaults = {row["id"]: row for row in self._model_defaults["models"]}
        model = defaults.get(model_id)
        if model is None:
            raise ReviewWorkspaceError("Unknown model")
        source_url = model.get("terms_source_url")
        parsed = urllib.parse.urlsplit(str(source_url))
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or parsed.query or parsed.fragment:
            raise ReviewWorkspaceError("The pinned model page is not a safe official HTTPS URL")
        request = urllib.request.Request(str(source_url), headers={"User-Agent": "drawable-local-review/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            final = urllib.parse.urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "huggingface.co":
                raise ReviewWorkspaceError("The model terms page redirected outside its pinned host")
            body = response.read(MAX_TERMS_BYTES + 1)
            if not body or len(body) > MAX_TERMS_BYTES:
                raise ReviewWorkspaceError("The pinned model page is empty or larger than 10 MiB")
            content_type = response.headers.get("Content-Type", "text/html").split(";", 1)[0]
            final_url = response.geturl()
        digest = hashlib.sha256(body).hexdigest()
        destination = self.root / "manifests" / "model-terms" / model_id / f"{digest}.html"
        if destination.is_symlink():
            raise ReviewWorkspaceError("Model terms evidence may not be a symbolic link")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not destination.exists():
            destination.write_bytes(body)
            os.chmod(destination, 0o600)
        elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ReviewWorkspaceError("Existing model terms evidence failed its content hash")
        receipt = {
            "schema_version": 1, "artifact_type": "drawable-model-terms-snapshot",
            "model_id": model_id, "purpose": "inference", "source_url": source_url,
            "captured_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "content_type": content_type, "content_sha256": digest,
            "content_base64": base64.b64encode(body).decode("ascii"),
        }
        receipt_bytes = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_path = self.root / "manifests" / "model-terms" / model_id / f"{receipt_digest}.json"
        if not receipt_path.exists():
            receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = receipt_path.with_name(f".{receipt_path.name}.{uuid4().hex}.tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(receipt_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, receipt_path)
            finally:
                temporary.unlink(missing_ok=True)
        self._secure_private_file(destination)
        self._secure_private_file(receipt_path)
        return {
            "status": "captured_pending_owner_review", "kind": "model", "model_id": model_id, "purpose": "inference",
            "source_url": source_url, "final_url": final_url,
            "snapshot_path": receipt_path.relative_to(self.root).as_posix(),
            "terms_snapshot_sha256": receipt_digest, "source_content_sha256": digest,
            "content_bytes": len(body),
            "captured_at": receipt["captured_at"], "network_used": True,
            "model_loaded": False, "training_started": False,
        }

    def record_decision(
        self,
        *,
        kind: Literal["source", "work", "model"],
        identifier: str,
        expected_revision: int,
        reviewer_id: str,
        decision: Literal["approved", "rejected"],
        terms_sha256: str,
        note: str,
        permissions: dict[str, bool] | None = None,
        attribution_text: str | None = None,
    ) -> dict[str, Any]:
        if not REVIEWER_PATTERN.fullmatch(reviewer_id):
            raise ReviewWorkspaceError("Enter a pseudonymous reviewer ID (3–80 letters, numbers, _ or -)")
        if not DIGEST_PATTERN.fullmatch(terms_sha256):
            raise ReviewWorkspaceError("The captured terms copy is missing or has an invalid hash")
        if len(note.strip()) < 12 or len(note) > 1000:
            raise ReviewWorkspaceError("Explain what you reviewed in a note of 12–1000 characters")
        with self._lock:
            state = self._state()
            if state["revision"] != expected_revision:
                raise ReviewWorkspaceError("Workspace changed; refresh before saving your decision")
            row_kind, _, current_digest = self._snapshot(kind, identifier)
            if row_kind != "source" and kind == "work":
                raise ReviewWorkspaceError("The work does not have selected parent-source terms")
            if current_digest != terms_sha256:
                raise ReviewWorkspaceError("The captured terms copy changed; inspect it again before deciding")
            if kind == "source":
                known = any(item["id"] == identifier for item in self._source_defaults["sources"])
                target = state["source_decisions"]
            elif kind == "work":
                known = any(item["folder_id"] == identifier for item in self._work_defaults["works"])
                known = known and identifier in state["selection"]["works"]
                target = state["work_decisions"]
            else:
                known = any(item["id"] == identifier for item in self._model_defaults["models"])
                target = state["model_decisions"]
            if not known:
                raise ReviewWorkspaceError("This item is not in the current review registry")
            permission_set = _permissions(permissions) if kind != "model" and decision == "approved" else dict.fromkeys(PERMISSIONS, False)
            if kind == "work" and decision == "approved":
                parent = state["source_decisions"].get(DEFAULT_SOURCE_SELECTION[0])
                source_allowed = parent.get("permissions", {}) if parent and parent.get("decision") == "approved" else {}
                if any(flag and not source_allowed.get(name, False) for name, flag in permission_set.items()):
                    raise ReviewWorkspaceError("A work cannot receive a use the owner declined for its source")
                if not attribution_text or len(attribution_text.strip()) < 8 or len(attribution_text) > 1000:
                    raise ReviewWorkspaceError("Approved work needs its reviewed attribution text")
                official = next(item for item in self._work_defaults["works"] if item["folder_id"] == identifier)
                if attribution_text.strip() != str(official.get("artist", "")).strip() and official.get("artist") not in attribution_text:
                    raise ReviewWorkspaceError("Attribution must include the artist named in the source registry")
            if kind == "model" and permissions is not None:
                raise ReviewWorkspaceError("Model terms reviews do not accept source use permissions")
            if kind == "model":
                artifact = self.root / "manifests" / "model-terms" / identifier / f"{terms_sha256}.json"
                if artifact.is_symlink() or not artifact.is_file():
                    raise ReviewWorkspaceError("The exact pinned model terms snapshot is missing")
            if kind == "source":
                terms_receipt = intake.terms_snapshot_receipt_path(self.root, identifier, terms_sha256)
                if terms_receipt.is_symlink() or not terms_receipt.is_file():
                    raise ReviewWorkspaceError("The captured official source terms copy is missing")
            if decision == "rejected":
                permission_set = dict.fromkeys(PERMISSIONS, False)
            now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
            item: dict[str, Any] = {
                "decision": decision, "terms_snapshot_sha256": terms_sha256,
                "reviewer_id": reviewer_id, "reviewed_at": now, "note": note.strip(),
                "permissions": permission_set,
            }
            if kind == "work":
                item["attribution_text"] = attribution_text.strip() if attribution_text else None
            target[identifier] = item
            state["revision"] += 1
            state["reviewer_id"] = reviewer_id
            state["events"].append({
                "kind": kind, "id": identifier, "decision": decision,
                "terms_snapshot_sha256": terms_sha256, "reviewer_id": reviewer_id,
                "reviewed_at": now, "note": note.strip(), "permissions": permission_set,
            })
            self._save_profile(state)
            _write_private_json(self.root, self.decisions_path, state)
            self._materialize(state)
            return {"saved": True, "revision": state["revision"], "decision": item}

    def select_works(self, *, expected_revision: int, work_ids: list[str]) -> dict[str, Any]:
        with self._lock:
            state = self._state()
            if state["revision"] != expected_revision:
                raise ReviewWorkspaceError("Workspace changed; refresh before saving work selection")
            known = {row.get("folder_id") for row in self._work_defaults.get("works", [])}
            if not work_ids or len(work_ids) != len(set(work_ids)) or any(item not in known for item in work_ids):
                raise ReviewWorkspaceError("Select one or more listed works")
            state["selection"]["works"] = work_ids
            self._save_profile(state)
            state["revision"] += 1
            _write_private_json(self.root, self.decisions_path, state)
            self._materialize(state)
            return {"saved": True, "revision": state["revision"], "work_ids": work_ids}

    def grant_preview_batch(
        self, *, expected_revision: int, reviewer_id: str, limit: int
    ) -> dict[str, Any]:
        """Grant local display to one explicit batch; never label or admit it to a gallery."""
        if not REVIEWER_PATTERN.fullmatch(reviewer_id):
            raise ReviewWorkspaceError("Enter a pseudonymous reviewer ID before preparing previews")
        if limit < 1 or limit > MAX_CANDIDATE_BATCH:
            raise ReviewWorkspaceError("A preview batch must contain between 1 and 32 assets")
        source_decision, _ = self._source_and_work_ready()
        state = self._state()
        if state["revision"] != expected_revision:
            raise ReviewWorkspaceError("Workspace changed; refresh before preparing the next preview batch")
        selected_ids = [
            work_id for work_id in state["selection"]["works"]
            if work_id in state["work_decisions"]
            and state["work_decisions"][work_id]["decision"] == "approved"
            and state["work_decisions"][work_id]["permissions"].get("local_display") is True
        ]
        if not selected_ids:
            raise ReviewWorkspaceError("No selected work is approved for local image display")
        work_map = {row["folder_id"]: row for row in self.works()["works"]}
        work_decisions = state["work_decisions"]
        from .curation import CurationReviewRequest, review_asset
        from .database import connect

        db = connect(self.settings.metadata_db)
        try:
            rows = db.execute(
                "SELECT asset_id, source_id, revision, state, metadata_json FROM assets "
                "WHERE source_id = 'anita-dataset' AND state IN ('quarantined', 'candidate_pending_rights') "
                "ORDER BY asset_id"
            ).fetchall()
        finally:
            db.close()
        buckets: dict[str, list[Any]] = {work_id: [] for work_id in selected_ids}
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            work_id = metadata.get("work_id")
            if work_id not in buckets or metadata.get("candidate_type") != "sketch":
                continue
            if metadata.get("decoder_status") != "decoded" or metadata.get("review_status") == "approved":
                continue
            if metadata.get("use_permissions", {}).get("local_display") is True:
                continue
            buckets[work_id].append((row, metadata))
        selected_rows: list[tuple[Any, dict[str, Any]]] = []
        while len(selected_rows) < limit and any(buckets.values()):
            for work_id in selected_ids:
                if buckets[work_id] and len(selected_rows) < limit:
                    selected_rows.append(buckets[work_id].pop(0))
        if not selected_rows:
            raise ReviewWorkspaceError("No unreviewed native sketch candidates remain in the selected works")
        results = []
        source_terms_hash = source_decision["terms_snapshot_sha256"]
        for row, metadata in selected_rows:
            work_id = str(metadata["work_id"])
            work = work_decisions[work_id]
            permissions = dict.fromkeys(PERMISSIONS, False)
            permissions["local_display"] = True
            payload = CurationReviewRequest.model_validate(
                {
                    "expected_revision": row["revision"],
                    "reviewer_id": reviewer_id,
                    "idempotency_key": str(uuid4()),
                    "decision": "quarantine",
                    "rights_status": "approved",
                    "license_id": work_map[work_id]["license"],
                    "attribution_text": work["attribution_text"],
                    "terms_snapshot_sha256": source_terms_hash,
                    "use_permissions": permissions,
                    "labels": None,
                    "learning_split": "unassigned",
                    "note": "Owner-granted local preview only; content and research labels still require review.",
                }
            )
            saved = review_asset(
                self.settings, row["asset_id"], payload,
                registry_path=self.sources_path, anita_terms_path=self.works_path,
            )
            results.append(saved.asset_id)
        state = self._state()
        state["revision"] += 1
        state["reviewer_id"] = reviewer_id
        state["events"].append({
            "kind": "preview_batch", "asset_ids": results,
            "terms_snapshot_sha256": source_terms_hash,
            "reviewer_id": reviewer_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "assets_approved": False,
        })
        _write_private_json(self.root, self.decisions_path, state)
        return {
            "status": "preview_batch_ready", "count": len(results),
            "asset_ids": results, "assets_approved": False,
            "gallery_activated": False, "review_url": "/curate",
            "revision": state["revision"],
        }

    def _source_and_work_ready(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        state = self._state()
        selected_source = state["selection"]["sources"]
        if len(selected_source) != 1 or selected_source[0] != "anita-dataset":
            raise ReviewWorkspaceError("The current visual-review flow supports one selected Anita source")
        source_decision = state["source_decisions"].get("anita-dataset")
        if not source_decision or source_decision["decision"] != "approved" or not source_decision["permissions"].get("local_display"):
            raise ReviewWorkspaceError("Approve local display for the source before preparing image previews")
        selected_work_ids = state["selection"]["works"]
        work_decisions = state["work_decisions"]
        rows = []
        for work_id in selected_work_ids:
            work = work_decisions.get(work_id)
            if not work or work["decision"] != "approved" or not work["permissions"].get("local_display"):
                continue
            rows.append(work)
        if not rows:
            raise ReviewWorkspaceError("Approve at least one selected work for local display")
        return source_decision, rows

    def queue_preparation(self, *, expected_revision: int, reviewer_id: str) -> str:
        if not REVIEWER_PATTERN.fullmatch(reviewer_id):
            raise ReviewWorkspaceError("Enter a pseudonymous reviewer ID before preparing candidates")
        self._source_and_work_ready()
        with self._lock:
            state = self._state()
            if state["revision"] != expected_revision:
                raise ReviewWorkspaceError("Workspace changed; refresh before starting preparation")
            if state.get("preparation", {}).get("status") in {"queued", "running"}:
                raise ReviewWorkspaceError("Candidate preparation is already running")
            job_id = str(uuid4())
            state["preparation"] = {
                "status": "queued", "job_id": job_id,
                "message": "Preparing approved source images for review.",
                "phase": "queued", "assets_approved": False, "gallery_activated": False,
            }
            state["reviewer_id"] = reviewer_id
            state["revision"] += 1
            _write_private_json(self.root, self.decisions_path, state)
            return job_id

    def _preparation_phase(self, job_id: str, phase: str, message: str) -> None:
        with self._lock:
            state = self._state()
            current = state.get("preparation", {})
            if not isinstance(current, dict) or current.get("job_id") != job_id:
                raise ReviewWorkspaceError("Candidate preparation job was replaced")
            current.update(status="running", phase=phase, message=message)
            state["revision"] += 1
            _write_private_json(self.root, self.decisions_path, state)

    def prepare_candidates(self, *, job_id: str) -> dict[str, Any]:
        """Acquire/fingerprint/catalogue approved work candidates; never approves assets."""
        self._preparation_phase(job_id, "checking_archive", "Checking the official archive and terms evidence.")
        with self._lock:
            source_decision, _ = self._source_and_work_ready()
            terms_hash = source_decision["terms_snapshot_sha256"]
            selection_state = self._state()
            approved_work_ids = sorted(
                work_id for work_id in selection_state["selection"]["works"]
                if (decision := selection_state["work_decisions"].get(work_id))
                and decision.get("decision") == "approved"
                and decision.get("permissions", {}).get("local_display") is True
            )
        root = self.root
        source_id = "anita-dataset"
        works = self.works_path
        sources = self.sources_path
        source = intake.source_by_id(source_id, sources)
        archive_receipt = root / "manifests" / "source-receipts" / f"{source_id}.json"
        if not archive_receipt.is_file():
            partial = root / f"{source['archive_path']}.part"
            intake.acquire_archive(
                root, source_id, registry_path=sources,
                resume=partial.is_file() and not partial.is_symlink(),
            )
        self._preparation_phase(job_id, "extracting", "Safely extracting the verified archive into private storage.")
        extraction = intake.extract_verified_archive(root, source_id, registry_path=sources)
        if extraction.get("status") != "extracted_pending_rights_and_review":
            raise ReviewWorkspaceError("The archive extraction did not reach its quarantined state")
        reconciliation = intake.reconcile_archived_source(root, source_id, registry_path=sources)
        if reconciliation.get("clean_extraction_status") != "reconciled":
            raise ReviewWorkspaceError("The official archive did not match its clean extraction")
        extracted_content = root / str(extraction["relative_extraction_path"]) / "content"
        archive_prefix = reconciliation.get("configured_tree_archive_prefix_removed")
        if archive_prefix:
            extracted_content = extracted_content / str(archive_prefix)
        self._preparation_phase(job_id, "fingerprinting", "Checking image bytes and building a source manifest.")
        manifest = intake.build_tree_manifest(
            root, source_id, tree_path=extracted_content,
            anita_work_ids=set(approved_work_ids),
            registry_path=sources, anita_terms_path=works,
        )
        archive_sha256 = json.loads(archive_receipt.read_text(encoding="utf-8"))["sha256"]
        manifest["source_archive_sha256"] = archive_sha256
        manifest["source_reconciliation_report"] = reconciliation["report_path"]
        manifest_path = root / "manifests" / (
            f"{source_id}-review-tree-{manifest['tree_sha256'][:16]}-"
            f"{manifest['source_registry_sha256'][:8]}-{manifest['anita_work_terms_sha256'][:8]}.json"
        )
        if not manifest_path.exists():
            intake.write_json_atomic(manifest_path, manifest)
        self._preparation_phase(job_id, "cataloguing", "Adding unapproved candidates to the private review queue.")
        with self._lock:
            current_source, _ = self._source_and_work_ready()
            if current_source["terms_snapshot_sha256"] != terms_hash:
                raise ReviewWorkspaceError("Source permission changed during preparation; review it again")
            current_state = self._state()
            current_work_ids = sorted(
                work_id for work_id in current_state["selection"]["works"]
                if (decision := current_state["work_decisions"].get(work_id))
                and decision.get("decision") == "approved"
                and decision.get("permissions", {}).get("local_display") is True
            )
            if current_work_ids != approved_work_ids:
                raise ReviewWorkspaceError("Selected work permission changed during preparation; review it again")
        catalog = catalog_manifest(
            root, self.settings.metadata_db, manifest_path,
            registry_path=sources, anita_terms_path=works,
        )
        with self._lock:
            state = self._state()
            if state.get("preparation", {}).get("job_id") != job_id:
                raise ReviewWorkspaceError("Candidate preparation job was replaced")
            total = catalog.get("inserted", 0) + catalog.get("updated", 0) + catalog.get("preserved", 0)
            state["preparation"] = {
                "status": "candidates_catalogued_for_preview", "source_id": source_id,
                "archive_sha256": archive_sha256,
                "reconciliation_status": reconciliation["status"],
                "clean_extraction_status": reconciliation["clean_extraction_status"],
                "reconciliation_report": reconciliation["report_path"],
                "tree_sha256": manifest["tree_sha256"], "manifest_path": manifest_path.relative_to(root).as_posix(),
                "catalogued_assets": total, "selected_work_ids": approved_work_ids,
                "terms_snapshot_sha256": terms_hash, "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "assets_approved": False, "gallery_activated": False,
                "job_id": job_id,
            }
            state["revision"] += 1
            _write_private_json(root, self.decisions_path, state)
            return state["preparation"]


def run_review_preparation(workspace: ReviewWorkspace, job_id: str) -> dict[str, Any]:
    """Run the bounded, resumable source preparation in a background API task."""
    try:
        return workspace.prepare_candidates(job_id=job_id)
    except Exception as error:  # noqa: BLE001 - persist every preparation failure as a resumable job state
        with workspace._lock:
            state = workspace._state()
            state["preparation"] = {
                "status": "blocked", "message": str(error)[:1000],
                "job_id": job_id, "phase": "blocked", "resumable": True,
                "assets_approved": False, "gallery_activated": False,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            state["revision"] += 1
            _write_private_json(workspace.root, workspace.decisions_path, state)
        return state["preparation"]
