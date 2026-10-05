from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from drawable_research_contracts.readiness import (
    REQUIREMENTS_REVISION,
    GateReportV1,
    GateReportV2,
    attest_gate,
    validate_readiness,
)


def _fixture(root: Path, *, reported_count: int = 12) -> tuple[Path, Path]:
    source = root / "manifests" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"verified":true}\n', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = GateReportV2(
        schema_version=2,
        requirements_revision=REQUIREMENTS_REVISION,
        artifact_type="drawable-gate-report",
        gate_id="source_yield_review",
        producer="drawable-data-sample",
        produced_at=datetime.now(UTC),
        checks={
            "sample_identity_verified": True,
            "owner_judgments_complete": True,
            "review_time_measured": True,
        },
        measurements={
            "reviewed_candidates": reported_count,
            "median_review_seconds_per_candidate": 45,
            "useful_eye_crop_candidates": 3,
            "viewpoint_identifiable_candidates": 7,
            "ambiguous_viewpoint_candidates": 2,
        },
        inputs=[{"path": "manifests/source.json", "sha256": digest}],
        profile_hashes={"source_selection": "a" * 64},
        summary="Human judgments recorded against the fixed source-yield sample.",
    )
    report_path = root / "reports" / "source-yield-evidence.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return source, report_path


def test_readiness_only_accepts_an_attested_gate_report_with_bound_inputs(tmp_path: Path) -> None:
    source, report_path = _fixture(tmp_path)
    ledger, _ = attest_gate(
        tmp_path,
        "source_yield_review",
        "reports/source-yield-evidence.json",
        "owner-local",
        "Reviewed the sample identities, labels and recorded timing.",
        None,
    )
    ledger_path = tmp_path / "reports" / "readiness-ledger.json"
    ledger_path.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")

    readiness = validate_readiness(tmp_path, "m0", ledger_path)
    assert readiness["status"] == "blocked"
    assert "source_yield_review" in readiness["verified_evidence"]
    assert readiness["training_authorized"] is False
    assert readiness["training_started"] is False
    assert readiness["m3_complete"] is False
    assert readiness["preparation_complete"] is False
    assert readiness["execution_status"] == "blocked"
    assert "trainer_run_resume_command_not_implemented" in readiness["training_execution_blockers"]

    source.write_text('{"verified":false}\n', encoding="utf-8")
    changed_input = validate_readiness(tmp_path, "m0", ledger_path)
    assert "Input checksum changed" in changed_input["blockers"]["source_yield_review"]

    source.write_text('{"verified":true}\n', encoding="utf-8")

    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    tampered = validate_readiness(tmp_path, "m0", ledger_path)
    assert "source_yield_review" in tampered["blockers"]
    assert tampered["preparation_complete"] is False


def test_gate_attestation_checks_required_measurements_and_checks(tmp_path: Path) -> None:
    _, report_path = _fixture(tmp_path, reported_count=0)
    with pytest.raises(ValueError, match="reviewed_candidates"):
        attest_gate(
            tmp_path,
            "source_yield_review",
            "reports/source-yield-evidence.json",
            "owner-local",
            "Reviewed the sample identities, labels and recorded timing.",
            None,
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["checks"]["owner_judgments_complete"] = False
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="owner_judgments_complete"):
        attest_gate(
            tmp_path,
            "source_yield_review",
            "reports/source-yield-evidence.json",
            "owner-local",
            "Reviewed the sample identities, labels and recorded timing.",
            None,
        )


def test_schema_v1_gate_report_remains_historical(tmp_path: Path) -> None:
    source = tmp_path / "manifests" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"verified":true}\n', encoding="utf-8")
    report = GateReportV1(
        schema_version=1,
        artifact_type="drawable-gate-report",
        gate_id="source_yield_review",
        producer="legacy-test",
        produced_at=datetime.now(UTC),
        checks={
            "sample_identity_verified": True,
            "owner_judgments_complete": True,
            "review_time_measured": True,
        },
        measurements={
            "reviewed_candidates": 12,
            "median_review_seconds_per_candidate": 45,
            "useful_eye_crop_candidates": 3,
            "viewpoint_identifiable_candidates": 7,
            "ambiguous_viewpoint_candidates": 2,
        },
        inputs=[{"path": "manifests/source.json", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        summary="Legacy sample report.",
    )
    report_path = tmp_path / "reports" / "legacy.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="historical"):
        attest_gate(
            tmp_path,
            "source_yield_review",
            "reports/legacy.json",
            "owner-local",
            "Reviewed legacy evidence for historical reference only.",
            None,
        )


def test_readiness_rejects_evidence_path_escape_and_symlinks(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "drawable-gate-report",
        "gate_id": "source_yield_review",
        "producer": "drawable-data-sample",
        "produced_at": datetime.now(UTC).isoformat(),
        "checks": {
            "sample_identity_verified": True,
            "owner_judgments_complete": True,
            "review_time_measured": True,
        },
        "measurements": {"reviewed_candidates": 12},
        "inputs": [{"path": "../private.json", "sha256": "0" * 64}],
        "summary": "Invalid evidence fixture.",
    }
    with pytest.raises(ValidationError, match="Evidence paths"):
        GateReportV1.model_validate(payload)


def test_effort_gate_enforces_five_hour_checkpoint_and_ten_hour_stop(tmp_path: Path) -> None:
    source = tmp_path / "effort-log.json"
    source.write_text('{"active_minutes":300}\n', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report_path = tmp_path / "reports" / "effort.json"
    report_path.parent.mkdir()

    def write_report(
        minutes: int, *, checkpoint: bool = False, stopped: bool = False
    ) -> None:
        report = GateReportV2(
            schema_version=2,
            requirements_revision=REQUIREMENTS_REVISION,
            artifact_type="drawable-gate-report",
            gate_id="human_effort_storage",
            producer="drawable-effort-report",
            produced_at=datetime.now(UTC),
            checks={
                "active_effort_measured": True,
                "five_hour_checkpoint_completed": checkpoint,
                "storage_headroom_checked": True,
                "annotation_stopped_at_budget": stopped,
            },
            measurements={
                "pilot_annotation_minutes": minutes,
                "disk_free_gib": 308,
                "projected_review_hours_for_full_program": 140,
            },
            profile_hashes={"source_selection": digest},
            inputs=[{"path": "effort-log.json", "sha256": digest}],
            summary="Owner's timed pilot-review and storage report.",
        )
        report_path.write_text(report.model_dump_json(), encoding="utf-8")

    def attest() -> None:
        attest_gate(
            tmp_path,
            "human_effort_storage",
            "reports/effort.json",
            "owner-local",
            "Reviewed active minutes, the checkpoint and storage headroom.",
            None,
        )

    write_report(0)
    attest()
    write_report(299)
    attest()
    write_report(300)
    with pytest.raises(ValueError, match="five-hour annotation checkpoint"):
        attest()
    write_report(300, checkpoint=True)
    attest()
    write_report(600, checkpoint=True)
    with pytest.raises(ValueError, match="hard stop"):
        attest()
    write_report(600, checkpoint=True, stopped=True)
    attest()
    write_report(601, stopped=True)
    with pytest.raises(ValueError, match="pilot_annotation_minutes"):
        attest()


def test_attestation_rejects_untrusted_raw_report_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="normalized and beneath"):
        attest_gate(
            tmp_path,
            "source_yield_review",
            "../outside.json",
            "owner-local",
            "Reviewed the sample identities, labels and recorded timing.",
            None,
        )
