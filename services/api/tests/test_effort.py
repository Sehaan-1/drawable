from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from drawable_api.effort import project_effort_plan
from drawable_api.intake import REPOSITORY_ROOT
from drawable_api.m0 import _effort_protocol_blockers


def _plan() -> dict:
    return {
        "schema_version": 1,
        "requirements_revision": "2026-10-04-practical-r1",
        "plan_id": "synthetic-effort-fixture",
        "enabled_assets_target": 20,
        "reference_strata": [
            {
                "stratum_id": "manga_anime/eye",
                "style": "manga_anime",
                "scope": "eye",
                "gold_target": 10,
                "reviewed_candidates": 10,
                "eligible_candidates": 5,
                "total_preparation_review_seconds": 200,
                "yield_low": 0.4,
                "yield_base": 0.5,
                "yield_high": 0.6,
                "seconds_low": 15,
                "seconds_high": 30,
                "source": "measured",
                "sample_size": 10,
                "evidence_path": "reports/private/stratum-review.json",
            }
        ],
        "drawing_rates": [
            {
                "rate_key": "freehand/standard",
                "seconds_low": 60,
                "seconds_base": 120,
                "seconds_high": 180,
                "source": "owner_estimate",
                "basis": "Estimate from current short freehand recording tasks.",
            }
        ],
        "drawings": [
            {
                "drawing_id": "drawing-a",
                "split": "training",
                "query_class": "in_domain",
                "task": "freehand",
                "complexity": "standard",
                "rate_key": "freehand/standard",
                "reused_evidence": ["m0-recording-a"],
            },
            {
                "drawing_id": "drawing-b",
                "split": "model_selection",
                "query_class": "in_domain",
                "task": "freehand",
                "complexity": "standard",
                "rate_key": "freehand/standard",
                "reused_evidence": [],
            },
        ],
        "judgment_pool": [
            {"query_id": "q-train", "candidate_id": "asset-a", "system_id": "mobileclip", "purpose": "training"},
            {"query_id": "q-train", "candidate_id": "asset-a", "system_id": "comparator", "purpose": "training"},
            {"query_id": "q-train", "candidate_id": "asset-b", "system_id": "mobileclip", "purpose": "training"},
            {"query_id": "q-final", "candidate_id": "asset-c", "system_id": "frozen-final", "purpose": "final"},
            {"query_id": "q-final", "candidate_id": "asset-c", "system_id": "ablation-final", "purpose": "final"},
        ],
        "judgment_rates": [
            {
                "purpose": "training",
                "seconds_low": 10,
                "seconds_base": 20,
                "seconds_high": 30,
                "source": "owner_estimate",
                "basis": "Estimate for blinded training-pair review.",
            },
            {
                "purpose": "final",
                "seconds_low": 15,
                "seconds_base": 30,
                "seconds_high": 45,
                "source": "owner_estimate",
                "basis": "Estimate for blinded final-pool pair review.",
            },
        ],
        "second_reviews": [
            {
                "query_id": "q-train",
                "candidate_id": "asset-a",
                "seconds_low": 10,
                "seconds_base": 20,
                "seconds_high": 30,
                "source": "owner_estimate",
                "basis": "Estimate a possible second independent review.",
            }
        ],
        "adjudications": [
            {
                "query_id": "q-train",
                "candidate_id": "asset-a",
                "seconds_low": 5,
                "seconds_base": 10,
                "seconds_high": 15,
                "source": "owner_estimate",
                "basis": "Estimate a short disagreement adjudication.",
            }
        ],
        "non_gold_acceptance": {
            "mode": "manual_review",
            "seconds_low": 5,
            "seconds_base": 10,
            "seconds_high": 15,
            "source": "owner_estimate",
            "basis": "Assume manual approval review for each enabled non-gold asset.",
        },
        "remaining_work": [
            {
                "category": category,
                "hours_low": 1,
                "hours_base": 2,
                "hours_high": 3,
                "source": "owner_estimate",
                "basis": f"Estimate remaining {category.replace('_', ' ')} work.",
            }
            for category in (
                "rights_and_source_acquisition",
                "indexing_preparation",
                "reviewer_coordination",
                "administration",
            )
        ],
        "storage_bytes": {
            "downloads": 100,
            "expanded_sources": 200,
            "derivatives": 300,
            "models_and_indexes": 400,
            "temporary_builds": 500,
        },
    }


def test_effort_plan_counts_unique_drawings_and_deduplicates_candidate_pools() -> None:
    projection, blockers = project_effort_plan(_plan())

    assert blockers == []
    assert projection["unique_drawings"] == 2
    assert projection["drawings_reusing_m0_or_pilot_evidence"] == 1
    assert projection["accepted_gold_target"] == 10
    assert projection["non_gold_enabled_assets"] == 10
    assert projection["development_calibration_judgment_pairs"] == 2
    assert projection["final_unique_query_candidate_pairs"] == 1
    assert projection["judgment_pairs_by_purpose"]["training"]["systems_in_union"] == [
        "comparator",
        "mobileclip",
    ]
    assert projection["projected_storage_bytes"] == 1500
    assert projection["projected_full_program_hours"]["low"] < projection[
        "projected_full_program_hours"
    ]["base"] < projection["projected_full_program_hours"]["high"]


def test_zero_yield_reference_stratum_blocks_projection() -> None:
    plan = deepcopy(_plan())
    stratum = plan["reference_strata"][0]
    stratum.update(eligible_candidates=0, yield_low=0, yield_base=0, yield_high=0)

    projection, blockers = project_effort_plan(plan)

    assert "reference_stratum_zero_yield:manga_anime/eye" in blockers
    assert "reference_stratum_low_yield_unresolved:manga_anime/eye" in blockers
    assert "reference_stratum_high_yield_unresolved:manga_anime/eye" in blockers
    assert projection["reference_strata"][0]["projected_candidate_reviews"] is None


def test_effort_plan_rejects_repeated_drawing_ids_and_cross_purpose_pairs() -> None:
    plan = _plan()
    plan["drawings"].append(deepcopy(plan["drawings"][0]))
    try:
        project_effort_plan(plan)
    except ValueError as error:
        assert "listed more than once" in str(error)
    else:
        raise AssertionError("Duplicate drawing IDs must fail")

    plan = _plan()
    plan["judgment_pool"].append(
        {"query_id": "q-train", "candidate_id": "asset-a", "system_id": "final", "purpose": "final"}
    )
    try:
        project_effort_plan(plan)
    except ValueError as error:
        assert "more than one protected purpose" in str(error)
    else:
        raise AssertionError("Cross-purpose judgment reuse must fail")


def test_effort_protocol_binds_full_counts_and_reconciles_replay_reuse() -> None:
    protocol_path = REPOSITORY_ROOT / Path("config/recipes/full-program-effort-v3.json")
    protocol = json.loads(protocol_path.read_bytes())
    reference_strata = [
        {"stratum_id": f"{style}/{scope}", "gold_target": target}
        for style, scopes in protocol["reference_gold_targets"].items()
        for scope, target in scopes.items()
    ]
    drawings = []
    for target in protocol["query_drawing_targets"]:
        for index in range(target["count"]):
            drawings.append(
                {
                    "drawing_id": f"{target['partition']}-{target['query_class']}-{index}",
                    "split": target["partition"],
                    "query_class": target["query_class"],
                }
            )
    replay_ids = {f"recording-{index}" for index in range(protocol["m0_feasibility_recordings"])}
    drawings.extend(
        {
            "drawing_id": drawing_id,
            "split": "pilot",
            "query_class": "in_domain",
            "reused_evidence": [drawing_id],
        }
        for drawing_id in sorted(replay_ids)[:5]
    )
    drawings.extend(
        {
            "drawing_id": f"pilot-{index}",
            "split": "pilot",
            "query_class": "in_domain",
        }
        for index in range(protocol["pilot_development_drawings"] - 5)
    )
    drawings.extend(
        {
            "drawing_id": drawing_id,
            "split": "m0",
            "query_class": "m0_feasibility",
        }
        for drawing_id in sorted(replay_ids)[5:]
    )
    plan = {
        "protocol_id": protocol["protocol_id"],
        "enabled_assets_target": protocol["enabled_assets_target"],
        "screened_candidates_target": protocol["screened_candidates_minimum"],
        "reference_strata": reference_strata,
        "drawings": drawings,
    }
    projection = {
        "judgment_pairs_by_purpose": {
            purpose: {"unique_query_candidate_pairs": count}
            for purpose, count in protocol["judgment_pair_minimums"].items()
        }
    }

    assert _effort_protocol_blockers(plan, protocol, projection, replay_ids) == []

    plan["screened_candidates_target"] -= 1
    assert "screened_candidate_target_is_below_the_authoritative_minimum" in (
        _effort_protocol_blockers(plan, protocol, projection, replay_ids)
    )
