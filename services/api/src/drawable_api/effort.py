"""Transparent, input-driven collection and judgment effort projections."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from drawable_research_contracts.readiness import REQUIREMENTS_REVISION

JUDGMENT_PURPOSES = ("training", "model_selection", "threshold", "audit", "final")
REMAINING_WORK_CATEGORIES = (
    "rights_and_source_acquisition",
    "indexing_preparation",
    "reviewer_coordination",
    "administration",
)
STORAGE_CATEGORIES = (
    "downloads",
    "expanded_sources",
    "derivatives",
    "models_and_indexes",
    "temporary_builds",
)


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _whole(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _range(
    row: dict[str, Any],
    prefix: str,
    *,
    maximum: float | None = None,
    minimum: float = 0.0,
) -> tuple[float, float, float]:
    values = tuple(
        _number(row.get(f"{prefix}_{part}"), f"{prefix}_{part}", minimum=minimum)
        for part in ("low", "base", "high")
    )
    low, base, high = values
    if low > base or base > high or (maximum is not None and high > maximum):
        raise ValueError(f"{prefix} must satisfy low <= base <= high")
    return low, base, high


def _rate_map(rows: Any, key_name: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a non-empty array")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key_name), str) or not row[key_name]:
            raise ValueError(f"Each {label} row needs a non-empty {key_name}")
        key = row[key_name]
        if key in result:
            raise ValueError(f"Duplicate {label} key: {key}")
        _range(row, "seconds", minimum=0.000001)
        if row.get("source") not in {"measured", "owner_estimate"}:
            raise ValueError(f"{label} source must be measured or owner_estimate")
        if row["source"] == "measured":
            _whole(row.get("sample_size"), f"{label} sample_size", minimum=1)
            if not isinstance(row.get("evidence_path"), str) or not row["evidence_path"]:
                raise ValueError(f"Measured {label} rows need an evidence_path")
        elif not isinstance(row.get("basis"), str) or len(row["basis"].strip()) < 12:
            raise ValueError(f"Owner-estimated {label} rows need a specific basis")
        result[key] = row
    return result


def project_effort_plan(plan: Any) -> tuple[dict[str, Any], list[str]]:
    """Calculate low/base/high hours from unique plan identities and explicit rates.

    Missing or zero-yield reference strata remain visible as blockers. Candidate
    pools are deduplicated by query and asset across systems before their review cost
    is estimated.
    """
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != 1
        or plan.get("requirements_revision") != REQUIREMENTS_REVISION
        or not isinstance(plan.get("plan_id"), str)
        or not plan["plan_id"].strip()
    ):
        raise ValueError("Effort plan schema, requirements revision, or plan_id is invalid")

    blockers: list[str] = []

    reference_strata = plan.get("reference_strata")
    if not isinstance(reference_strata, list) or not reference_strata:
        raise ValueError("reference_strata must be a non-empty array")
    seen_strata: set[str] = set()
    gold_target = 0
    reference_counts: list[dict[str, Any]] = []
    reference_hours = {"low": 0.0, "base": 0.0, "high": 0.0}
    for row in reference_strata:
        if not isinstance(row, dict) or not isinstance(row.get("stratum_id"), str) or not row["stratum_id"]:
            raise ValueError("Each reference stratum needs a stable stratum_id")
        name = row["stratum_id"]
        if name in seen_strata:
            raise ValueError(f"Duplicate reference stratum: {name}")
        seen_strata.add(name)
        target = _whole(row.get("gold_target"), f"{name}.gold_target", minimum=1)
        reviewed = _whole(row.get("reviewed_candidates"), f"{name}.reviewed_candidates")
        eligible = _whole(row.get("eligible_candidates"), f"{name}.eligible_candidates")
        if eligible > reviewed:
            raise ValueError(f"{name}.eligible_candidates cannot exceed reviewed_candidates")
        total_seconds = _number(
            row.get("total_preparation_review_seconds"),
            f"{name}.total_preparation_review_seconds",
        )
        if row.get("source") not in {"measured", "owner_estimate"}:
            raise ValueError(f"{name}.source must be measured or owner_estimate")
        if row["source"] == "measured":
            _whole(row.get("sample_size"), f"{name}.sample_size", minimum=1)
            if not isinstance(row.get("evidence_path"), str) or not row["evidence_path"]:
                raise ValueError(f"Measured reference stratum {name} needs an evidence_path")
        elif not isinstance(row.get("basis"), str) or len(row["basis"].strip()) < 12:
            raise ValueError(f"Owner-estimated reference stratum {name} needs a specific basis")
        yield_range = _range(row, "yield", maximum=1.0)
        if reviewed == 0:
            blockers.append(f"reference_stratum_unmeasured:{name}")
            mean_seconds = 0.0
            base_yield = 0.0
        else:
            mean_seconds = total_seconds / reviewed
            base_yield = eligible / reviewed
        low_seconds = _number(row.get("seconds_low"), f"{name}.seconds_low", minimum=0.000001)
        high_seconds = _number(row.get("seconds_high"), f"{name}.seconds_high", minimum=0.000001)
        if reviewed > 0 and (low_seconds > mean_seconds or mean_seconds > high_seconds):
            raise ValueError(f"{name} seconds must satisfy low <= measured mean <= high")
        low_yield, _, high_yield = yield_range
        if reviewed > 0 and not math.isclose(yield_range[1], base_yield, abs_tol=1e-9):
            raise ValueError(f"{name}.yield_base must equal eligible_candidates / reviewed_candidates")
        expected_reviews: int | None = None
        estimates: dict[str, float | None] = {"low": None, "base": None, "high": None}
        if eligible == 0 or base_yield == 0:
            blockers.append(f"reference_stratum_zero_yield:{name}")
            if low_yield <= 0:
                blockers.append(f"reference_stratum_low_yield_unresolved:{name}")
            if high_yield <= 0:
                blockers.append(f"reference_stratum_high_yield_unresolved:{name}")
        else:
            expected_reviews = math.ceil(target / base_yield)
            estimates["base"] = expected_reviews * mean_seconds / 3600
            if low_yield <= 0:
                blockers.append(f"reference_stratum_low_yield_unresolved:{name}")
            else:
                estimates["high"] = math.ceil(target / low_yield) * high_seconds / 3600
            if high_yield <= 0:
                blockers.append(f"reference_stratum_high_yield_unresolved:{name}")
            else:
                estimates["low"] = math.ceil(target / high_yield) * low_seconds / 3600
        for scenario in reference_hours:
            amount = estimates[scenario]
            if amount is not None:
                reference_hours[scenario] += amount
        gold_target += target
        reference_counts.append(
            {
                "stratum_id": name,
                "style": row.get("style"),
                "scope": row.get("scope"),
                "gold_target": target,
                "reviewed_candidates": reviewed,
                "eligible_candidates": eligible,
                "observed_yield": base_yield,
                "mean_preparation_review_seconds": mean_seconds,
                "projected_candidate_reviews": expected_reviews,
                "projected_hours": estimates,
                "rate_source": row["source"],
            }
        )

    enabled_target = _whole(plan.get("enabled_assets_target"), "enabled_assets_target", minimum=1)
    if gold_target > enabled_target:
        raise ValueError("Sum of accepted gold targets cannot exceed enabled_assets_target")

    drawing_rates = _rate_map(plan.get("drawing_rates"), "rate_key", "drawing_rates")
    drawings = plan.get("drawings")
    if not isinstance(drawings, list) or not drawings:
        raise ValueError("drawings must be a non-empty array of unique drawing identities")
    drawing_ids: set[str] = set()
    drawings_by_rate: dict[str, int] = defaultdict(int)
    drawing_partitions: dict[str, int] = defaultdict(int)
    reused_drawing_ids: list[str] = []
    for row in drawings:
        if not isinstance(row, dict) or not isinstance(row.get("drawing_id"), str) or not row["drawing_id"]:
            raise ValueError("Each drawing plan row needs a stable drawing_id")
        drawing_id = row["drawing_id"]
        if drawing_id in drawing_ids:
            raise ValueError(f"Drawing {drawing_id} is listed more than once; reuse it in one row")
        drawing_ids.add(drawing_id)
        split = row.get("split")
        if split not in {"training", "model_selection", "threshold", "calibration_audit", "locked_test", "pilot", "m0"}:
            raise ValueError(f"Drawing {drawing_id} has an unsupported split")
        query_class = row.get("query_class")
        if query_class not in {"in_domain", "unknown", "m0_feasibility"}:
            raise ValueError(f"Drawing {drawing_id} needs a supported query_class")
        if (split == "m0") != (query_class == "m0_feasibility"):
            raise ValueError(f"Drawing {drawing_id} query_class does not match its split")
        if split == "pilot" and query_class != "in_domain":
            raise ValueError(f"Pilot drawing {drawing_id} must be in-domain development data")
        rate_key = row.get("rate_key")
        if rate_key not in drawing_rates:
            raise ValueError(f"Drawing {drawing_id} has no defined rate_key")
        drawings_by_rate[str(rate_key)] += 1
        drawing_partitions[str(split)] += 1
        reuse = row.get("reused_evidence", [])
        if (
            not isinstance(reuse, list)
            or any(not isinstance(item, str) or not item for item in reuse)
            or len(reuse) != len(set(reuse))
        ):
            raise ValueError(f"Drawing {drawing_id}.reused_evidence must be an array of evidence IDs")
        if reuse:
            reused_drawing_ids.append(drawing_id)
    drawing_hours = {"low": 0.0, "base": 0.0, "high": 0.0}
    for rate_key, count in drawings_by_rate.items():
        rate = drawing_rates[rate_key]
        for scenario in drawing_hours:
            drawing_hours[scenario] += count * float(rate[f"seconds_{scenario}"]) / 3600

    pool_rows = plan.get("judgment_pool")
    if not isinstance(pool_rows, list) or not pool_rows:
        raise ValueError("judgment_pool must list query/candidate/system rows")
    pools: dict[str, dict[tuple[str, str], set[str]]] = {
        purpose: defaultdict(set) for purpose in JUDGMENT_PURPOSES
    }
    pair_purpose: dict[tuple[str, str], str] = {}
    for row in pool_rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("query_id"), str)
            or not row["query_id"]
            or not isinstance(row.get("candidate_id"), str)
            or not row["candidate_id"]
            or not isinstance(row.get("system_id"), str)
            or not row["system_id"]
            or row.get("purpose") not in JUDGMENT_PURPOSES
        ):
            raise ValueError("judgment_pool rows need query_id, candidate_id, system_id, and a supported purpose")
        purpose = str(row["purpose"])
        pair = (str(row["query_id"]), str(row["candidate_id"]))
        other_purpose = pair_purpose.setdefault(pair, purpose)
        if other_purpose != purpose:
            raise ValueError(f"Judgment pair {pair} appears in more than one protected purpose")
        pools[purpose][pair].add(str(row["system_id"]))

    judgment_rates = _rate_map(plan.get("judgment_rates"), "purpose", "judgment_rates")
    if set(judgment_rates) - set(JUDGMENT_PURPOSES):
        raise ValueError("judgment_rates contains an unsupported purpose")
    judgment_counts = {purpose: len(pools[purpose]) for purpose in JUDGMENT_PURPOSES}
    for purpose, count in judgment_counts.items():
        if count == 0:
            continue
        judgment_rate = judgment_rates.get(purpose)
        if judgment_rate is None:
            raise ValueError(f"judgment_rates is missing a rate for {purpose}")

    def ancillary_hours(field: str) -> tuple[dict[str, float], int]:
        rows = plan.get(field)
        if not isinstance(rows, list):
            raise TypeError(f"{field} must be an array")
        seen: set[tuple[str, str]] = set()
        totals = {"low": 0.0, "base": 0.0, "high": 0.0}
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"{field} rows must be objects")
            pair = (str(row.get("query_id", "")), str(row.get("candidate_id", "")))
            if not all(pair) or pair in seen or pair not in pair_purpose:
                raise ValueError(f"{field} must name unique pairs from judgment_pool")
            seen.add(pair)
            if row.get("source") not in {"measured", "owner_estimate"}:
                raise ValueError(f"{field} rows need source measured or owner_estimate")
            if row["source"] == "measured":
                if not isinstance(row.get("evidence_path"), str) or not row["evidence_path"]:
                    raise ValueError(f"Measured {field} rows need an evidence_path")
            elif not isinstance(row.get("basis"), str) or len(row["basis"].strip()) < 12:
                raise ValueError(f"Owner-estimated {field} rows need a specific basis")
            for scenario, seconds in zip(
                ("low", "base", "high"), _range(row, "seconds", minimum=0.000001)
            ):
                totals[scenario] += seconds / 3600
        return totals, len(seen)

    second_hours, second_review_count = ancillary_hours("second_reviews")
    adjudication_hours, adjudication_count = ancillary_hours("adjudications")

    non_gold = plan.get("non_gold_acceptance")
    if not isinstance(non_gold, dict) or non_gold.get("mode") not in {"manual_review", "human_audit"}:
        raise ValueError("non_gold_acceptance needs mode manual_review or human_audit")
    remaining_assets = enabled_target - gold_target
    if non_gold["mode"] == "manual_review":
        non_gold_count = remaining_assets
    else:
        non_gold_count = _whole(non_gold.get("audit_items"), "non_gold_acceptance.audit_items", minimum=1)
        if non_gold_count > remaining_assets:
            raise ValueError("Non-gold audit_items cannot exceed the planned non-gold enabled assets")
    non_gold_seconds = _range(non_gold, "seconds", minimum=0.000001)
    if non_gold.get("source") not in {"measured", "owner_estimate"}:
        raise ValueError("non_gold_acceptance source must be measured or owner_estimate")
    if non_gold["source"] == "owner_estimate" and not isinstance(non_gold.get("basis"), str):
        raise ValueError("Owner-estimated non-gold acceptance needs a basis")
    non_gold_hours = {
        scenario: non_gold_count * non_gold_seconds[index] / 3600
        for index, scenario in enumerate(("low", "base", "high"))
    }

    remaining_rows = plan.get("remaining_work")
    if not isinstance(remaining_rows, list):
        raise TypeError("remaining_work must be an array")
    remaining_by_category: dict[str, dict[str, Any]] = {}
    remaining_hours = {"low": 0.0, "base": 0.0, "high": 0.0}
    for row in remaining_rows:
        if not isinstance(row, dict) or not isinstance(row.get("category"), str):
            raise TypeError("remaining_work rows need a category")
        category = row["category"]
        if category in remaining_by_category:
            raise ValueError(f"Duplicate remaining_work category: {category}")
        if row.get("source") != "owner_estimate" or not isinstance(row.get("basis"), str) or len(row["basis"].strip()) < 12:
            raise ValueError(f"remaining_work {category} must be explicitly owner-estimated with a basis")
        hours = tuple(_number(row.get(f"hours_{part}"), f"{category}.hours_{part}") for part in ("low", "base", "high"))
        if hours[0] > hours[1] or hours[1] > hours[2]:
            raise ValueError(f"{category} hours must satisfy low <= base <= high")
        remaining_by_category[category] = row
        for part, amount in zip(("low", "base", "high"), hours):
            remaining_hours[part] += amount
    missing_categories = sorted(set(REMAINING_WORK_CATEGORIES) - set(remaining_by_category))
    if missing_categories:
        raise ValueError(f"remaining_work is missing categories: {', '.join(missing_categories)}")

    storage = plan.get("storage_bytes")
    if not isinstance(storage, dict):
        raise TypeError("storage_bytes must itemize the future storage estimate")
    missing_storage = sorted(set(STORAGE_CATEGORIES) - set(storage))
    if missing_storage:
        raise ValueError(f"storage_bytes is missing categories: {', '.join(missing_storage)}")
    storage_values = {key: _whole(storage[key], f"storage_bytes.{key}") for key in STORAGE_CATEGORIES}
    projected_storage_bytes = sum(storage_values.values())

    development_judgments = sum(judgment_counts[p] for p in JUDGMENT_PURPOSES if p != "final")
    judgment_by_purpose = {
        purpose: {
            "unique_query_candidate_pairs": judgment_counts[purpose],
            "systems_in_union": sorted(
                {
                    system
                    for system_set in pools[purpose].values()
                    for system in system_set
                }
            ),
        }
        for purpose in JUDGMENT_PURPOSES
    }

    total_hours = {"low": 0.0, "base": 0.0, "high": 0.0}
    remaining_total = {part: remaining_hours[part] + non_gold_hours[part] for part in remaining_hours}
    components = {
        "H_ref": reference_hours,
        "H_draw": drawing_hours,
        "H_development_judgments": {
            scenario: sum(
                judgment_counts[p] * float(judgment_rates[p][f"seconds_{scenario}"]) / 3600
                for p in JUDGMENT_PURPOSES
                if p != "final" and judgment_counts[p] and p in judgment_rates
            )
            for scenario in ("low", "base", "high")
        },
        "H_final_judgments": {
            scenario: judgment_counts["final"] * float(judgment_rates["final"][f"seconds_{scenario}"]) / 3600
            if judgment_counts["final"] and "final" in judgment_rates
            else 0.0
            for scenario in ("low", "base", "high")
        },
        "H_second_reviews": second_hours,
        "H_adjudication": adjudication_hours,
        "H_remaining": remaining_total,
    }
    for part in total_hours:
        total_hours[part] = sum(values[part] for values in components.values())

    projection = {
        "plan_id": plan["plan_id"],
        "enabled_assets_target": enabled_target,
        "accepted_gold_target": gold_target,
        "non_gold_enabled_assets": remaining_assets,
        "reference_strata": reference_counts,
        "unique_drawings": len(drawing_ids),
        "unique_drawing_partitions": dict(sorted(drawing_partitions.items())),
        "drawings_reusing_m0_or_pilot_evidence": len(reused_drawing_ids),
        "drawing_rate_strata": dict(sorted(drawings_by_rate.items())),
        "judgment_pairs_by_purpose": judgment_by_purpose,
        "development_calibration_judgment_pairs": development_judgments,
        "final_unique_query_candidate_pairs": judgment_counts["final"],
        "second_review_pairs": second_review_count,
        "adjudicated_pairs": adjudication_count,
        "non_gold_acceptance": {
            "mode": non_gold["mode"],
            "items_budgeted": non_gold_count,
            "hours": non_gold_hours,
            "basis": non_gold.get("basis"),
        },
        "hours_by_component": components,
        "projected_full_program_hours": total_hours,
        "storage_bytes_by_category": storage_values,
        "projected_storage_bytes": projected_storage_bytes,
        "scenarios_are_planning_ranges_not_confidence_intervals": True,
        "unresolved_strata": sorted(
            blocker.split(":", 1)[1]
            for blocker in blockers
            if blocker.startswith("reference_stratum_")
        ),
    }
    return projection, sorted(set(blockers))
