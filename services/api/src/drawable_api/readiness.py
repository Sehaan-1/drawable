from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from drawable_research_contracts.readiness import SPECS, STAGES, validate_readiness
from pydantic import BaseModel, ConfigDict, Field

from .intake import audit_sources
from .settings import REPOSITORY_ROOT

GateStatus = Literal["complete", "incomplete", "not_measured"]


class ReadinessGateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: GateStatus
    detail: str
    evidence_paths: list[str] = Field(default_factory=list)


class TrainingReadinessV2(BaseModel):
    """Read-only preparation report; never grants training authorization."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[2] = 2
    generated_at: str
    project_stage: Literal["pre_training_preparation"] = "pre_training_preparation"
    m0_status: Literal["incomplete", "complete"]
    m1_status: Literal["incomplete", "complete"]
    m2_status: Literal["not_achieved", "complete"]
    m3_status: Literal["not_achieved", "complete"]
    gates: dict[str, ReadinessGateV1]
    source_archive_status: dict[str, str]
    emoreccom_annotation_status: str
    local_gpu_status: Literal["not_probed"] = "not_probed"
    preparation_status: Literal["blocked", "complete"]
    execution_status: Literal["blocked"] = "blocked"
    training_authorization: Literal["pending"] = Field(
        default="pending", alias="trainingAuthorization"
    )
    training_started: Literal[False] = Field(default=False, alias="trainingStarted")
    can_request_separate_authorization: bool = Field(
        default=False, alias="canRequestSeparateAuthorization"
    )


_OLD_GATE_NAMES = {
    "official_archives_and_tree_reconciliation": "official_archive_reconciliation",
    "source_and_work_terms": "source_work_rights_review",
    "fixed_source_yield_review": "source_yield_review",
    "m0_sequence_smoke_set": "ten_feasibility_sequences",
    "m0_feasibility_and_effort_report": "human_effort_storage",
    "approved_pilot_gallery": "small_approved_gallery",
    "frozen_retrieval_feasibility": "frozen_cpu_forward",
    "sequence_corpus_and_locked_splits": "full_query_corpus",
    "fresh_clone_contract_ci_and_offline_startup": "fresh_clone_report",
}


def build_training_readiness(
    data_root: Path,
    *,
    registry_path: Path | None = None,
    ledger_path: Path | None = None,
) -> TrainingReadinessV2:
    root = data_root.resolve()
    source_registry = registry_path or REPOSITORY_ROOT / "config" / "sources.json"
    audit = audit_sources(root, registry_path=source_registry, verify_archives=False)
    archive_status = {row["source_id"]: row["status"] for row in audit["sources"]}
    annotations = str(audit["emoreccom_annotations"].get("status", "not_measured"))
    if ledger_path is None:
        default_ledger = root / "reports" / "readiness-ledger-v2.json"
        ledger = default_ledger if default_ledger.is_file() else None
    else:
        ledger = ledger_path
    readiness = validate_readiness(root, "training_preparation", ledger)

    by_id = {spec.gate_id: spec for spec in SPECS}
    attested = readiness["verified_evidence"]
    blockers = readiness["blockers"]
    gates: dict[str, ReadinessGateV1] = {}
    for public_name, gate_id in _OLD_GATE_NAMES.items():
        if gate_id in attested:
            state: GateStatus = "complete"
            detail = f"Owner-reviewed, hash-verified evidence for {by_id[gate_id].title}."
            paths = [str(attested[gate_id]["report_path"])]
        else:
            reason = blockers.get(gate_id, "Owner-reviewed evidence has not been recorded.")
            state = "not_measured" if reason == "owner-reviewed_evidence_missing" else "incomplete"
            detail = str(reason)
            paths = []
        gates[public_name] = ReadinessGateV1(status=state, detail=detail, evidence_paths=paths)
    for gate_id in readiness["required_gates"]:
        if gate_id not in gates:
            is_complete = gate_id in attested
            gates[gate_id] = ReadinessGateV1(
                status="complete" if is_complete else "incomplete",
                detail=(
                    f"Owner-reviewed, hash-verified evidence for {by_id[gate_id].title}."
                    if is_complete
                    else str(
                        blockers.get(gate_id, "Owner-reviewed evidence has not been recorded.")
                    )
                ),
                evidence_paths=[str(attested[gate_id]["report_path"])] if is_complete else [],
            )

    complete_by_milestone = {
        "m0": all(gate in attested for gate in STAGES["m0"]),
        "m1": all(gate in attested for gate in STAGES["m1"]),
        "pilot": all(gate in attested for gate in STAGES["m2"]),
        "m3": bool(readiness["m3_complete"]),
    }
    preparation_complete = bool(readiness["preparation_complete"])
    return TrainingReadinessV2(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        m0_status="complete" if complete_by_milestone["m0"] else "incomplete",
        m1_status=(
            "complete"
            if complete_by_milestone["m0"] and complete_by_milestone["m1"]
            else "incomplete"
        ),
        m2_status=(
            "complete"
            if complete_by_milestone["m0"]
            and complete_by_milestone["m1"]
            and complete_by_milestone["pilot"]
            else "not_achieved"
        ),
        m3_status="complete" if complete_by_milestone["m3"] else "not_achieved",
        gates=gates,
        source_archive_status=archive_status,
        emoreccom_annotation_status=annotations,
        preparation_status="complete" if preparation_complete else "blocked",
        trainingAuthorization="pending",
        trainingStarted=False,
        canRequestSeparateAuthorization=preparation_complete,
    )
