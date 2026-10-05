from __future__ import annotations

from pathlib import Path
from typing import Any

from drawable_research_contracts.readiness import (
    SPECS,
    STAGES,
    GateAttestationV1,
    GateReportV2,
    ReadinessLedgerV1,
    attest_gate,
    validate_readiness,
)

from .io import MLError

REQUIRED_GATES: dict[str, tuple[str, ...]] = {
    stage: STAGES[stage]
    for stage in (
        "m0", "m1", "pilot", "m3", "m5", "m6", "m7", "training_preparation", "training_execution"
    )
}


def readiness_report(data_root: Path, stage: str, ledger_path: Path | None) -> dict[str, Any]:
    try:
        return validate_readiness(data_root, stage, ledger_path)
    except (OSError, TypeError, ValueError) as error:
        raise MLError(f"Readiness evidence validation failed: {error}") from error


__all__ = [
    "REQUIRED_GATES",
    "SPECS",
    "STAGES",
    "GateAttestationV1",
    "GateReportV2",
    "ReadinessLedgerV1",
    "attest_gate",
    "readiness_report",
]
