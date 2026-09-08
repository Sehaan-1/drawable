"""End-to-end tests for the curation engine, safety semantics included.

Covers:

* ``GET /curation/next`` — stratified queue + filter behaviour, queue-empty
  semantics, the gallery-required gate, **session cursor/skip semantics**
  (repeat Next/Skip always advance, skipped assets never return), and stable
  by-id retrieval via ``GET /curation/candidates/{asset_id}``.
* ``POST /curation/labels`` — validation, dual-write to ``curation_labels``
  and ``assets``, the ``assets.enabled`` flip, the in-memory ``state.assets``
  refresh, and **optimistic label versioning**: a stale
  ``expected_label_version`` is a structured 409 carrying reconciliation
  information, so two curators can never silently overwrite each other.
* SFW adjudication — the quarantined backlog is metadata-only, previewing a
  held record requires a deliberate, expiring reveal grant, the public
  asset routes never serve held records, and safe/unsafe adjudications move
  records between the quarantine and review flows atomically.
* Crops & derivatives — an immutable child with fresh files/hashes and its
  own processing/review state, boundary validation, failure/retry of the
  required processing, and re-hydration after a gallery rebuild (stale
  parent artifacts never survive, revoked parent grants propagate).
* ``POST /curation/snapshots`` — JSON file emission with exclusive
  publication (concurrent exports cannot overwrite), exact label↔snapshot
  linking, a recorded content hash, reproducibility after later edits, and
  the breakdown shape.
* ``GET /curation/progress`` — the ``by_style`` / ``by_scope`` fields and
  the quarantined backlog counter.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from linescout_ml.taxonomy import PrimaryStyle, ScopeLabel
from PIL import Image

from linescout_api.gallery import enabled_assets, sync_gallery
from linescout_api.main import create_app
from linescout_api.routers import curation as curation_module
from tests.conftest import SYNTHETIC_MANIFEST, make_settings

# ----------------------------------------------------------------- helpers


def _seed_files(tmp_path: Path) -> None:
    """Write a 16×16 PNG for every referenced asset path in the synthetic manifest."""
    manifest = json.loads(SYNTHETIC_MANIFEST.read_text(encoding="utf-8"))
    base = SYNTHETIC_MANIFEST.parent
    for record in manifest["records"]:
        for key in ("line_art_path", "thumbnail_path"):
            target = base / record[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(target, "PNG")


@pytest.fixture
def curation_client(tmp_path: Path):
    """Yield a TestClient whose lifespan has run (so ``app.state.linescout`` is built).

    The synthetic manifest ships every asset pre-accepted, which would leave
    the curation queue empty. We force every asset back to ``unreviewed`` in
    the database once the lifespan is up, so the queue logic is exercised
    end-to-end without touching the committed fixture.
    """
    _seed_files(tmp_path)
    with TestClient(create_app(make_settings(tmp_path, curation_mode=True))) as client:
        state = client.app.state.linescout
        if state.gallery is not None:
            state.connection.execute(
                "UPDATE assets SET review_state = 'unreviewed', review_quality = NULL,"
                " gold_member = 0, enabled = 0"
            )
            state.assets = enabled_assets(state.connection)
        yield client


@pytest.fixture
def crop_client(tmp_path: Path):
    """Like ``curation_client`` but the gallery lives in a writable copy.

    Crop derivatives are written under the gallery data root; the committed
    synthetic fixture directory must stay pristine, so this fixture copies
    the whole gallery (manifest + PNGs) into ``tmp_path`` first. Rebuild
    tests mutate the copied manifest.
    """
    gallery_root = tmp_path / "gallery"
    shutil.copytree(SYNTHETIC_MANIFEST.parent, gallery_root)
    with TestClient(
        create_app(
            make_settings(
                tmp_path, curation_mode=True, gallery_manifest=gallery_root / "manifest.json"
            )
        )
    ) as client:
        state = client.app.state.linescout
        state.connection.execute(
            "UPDATE assets SET review_state = 'unreviewed', review_quality = NULL,"
            " gold_member = 0, enabled = 0"
        )
        state.assets = enabled_assets(state.connection)
        yield client


def _state(client: TestClient):
    return client.app.state.linescout


def _label(candidate: dict | str, **overrides: object) -> dict[str, object]:
    """A valid label body derived from a live candidate (id, state, version)."""
    if isinstance(candidate, dict):
        payload: dict[str, object] = {
            "asset_id": candidate["asset_id"],
            "expected_review_state": candidate["review_state"],
            "expected_label_version": candidate["label_version"],
        }
    else:
        payload = {
            "asset_id": candidate,
            "expected_review_state": "unreviewed",
            "expected_label_version": 0,
        }
    payload.update(
        {
            "decision": "keep",
            "quality": 3,
            "blockers": [],
        }
    )
    payload.update(overrides)
    return payload


def _candidate(client: TestClient, asset_id: str) -> dict:
    """Live candidate by id — the stable read path used by Previous/reconcile."""
    response = client.get(f"/api/v1/curation/candidates/{asset_id}")
    assert response.status_code == 200
    return response.json()


def _hold(state, asset_id: str, *, review_state: str | None = "quarantined") -> None:
    """Manufacture a held record (the fixture ships no screening verdicts)."""
    state.connection.execute(
        "UPDATE assets SET review_state = ? WHERE asset_id = ?",
        (review_state, asset_id),
    )


def _crop_body(candidate: dict, crop: dict, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "crop": crop,
        "expected_label_version": candidate["label_version"],
    }
    payload.update(overrides)
    return payload


def _gallery_dir(client: TestClient) -> Path:
    gallery = _state(client).gallery
    assert gallery is not None
    return gallery.data_root


def _data_dir(client: TestClient) -> Path:
    return Path(_state(client).settings.data_dir)


# ----------------------------------------------------------------- progress


def test_progress_breakdowns_match_assets_table(curation_client: TestClient) -> None:
    body = curation_client.get("/api/v1/curation/progress").json()
    assert body["reviewed"] == 0
    assert body["accepted"] == 0
    assert body["rejected"] == 0
    assert body["target"] == 2000
    assert body["remaining"] == 2000 - 0
    # Nothing is held at the start: the fixture has no screening verdicts and
    # no human flags survive the unreviewed reset.
    assert body["quarantined"] == 0

    # Every style / scope bucket must appear. No work has been done, so
    # ``reviewed``/``accepted``/``rejected`` are zero everywhere; ``remaining``
    # matches the live count of unreviewed assets in that bucket.
    assert set(body["by_style"]) == {s.value for s in PrimaryStyle}
    assert set(body["by_scope"]) == {s.value for s in ScopeLabel if s is not ScopeLabel.UNKNOWN}
    for payload in body["by_style"].values():
        assert payload["reviewed"] == 0
        assert payload["accepted"] == 0
        assert payload["rejected"] == 0
    for payload in body["by_scope"].values():
        assert payload["reviewed"] == 0
        assert payload["accepted"] == 0
        assert payload["rejected"] == 0

    # The remaining counts must equal the live unreviewed-asset count in the
    # database for every style bucket. Pull those counts directly so the test
    # is independent of fixture content.
    state = curation_client.app.state.linescout
    for style in PrimaryStyle:
        row = state.connection.execute(
            "SELECT COUNT(*) AS n FROM assets"
            " WHERE primary_style = ? AND review_state = 'unreviewed'",
            (style.value,),
        ).fetchone()
        assert body["by_style"][style.value]["remaining"] == int(row["n"])


def test_progress_counts_held_records(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    _hold(_state(curation_client), first["asset_id"])
    body = curation_client.get("/api/v1/curation/progress").json()
    assert body["quarantined"] == 1


# ----------------------------------------------------------------- next (no session)


def test_next_returns_stratified_candidate(curation_client: TestClient) -> None:
    body = curation_client.get("/api/v1/curation/next").json()
    assert "asset_id" in body
    assert body["primary_style"] in {s.value for s in PrimaryStyle}
    assert body["review_state"] == "unreviewed"
    # The wire response must always expose the asset URLs the UI needs, and
    # the per-asset curation version every subsequent write must echo back.
    assert body["thumbnail_url"].startswith("/api/v1/curation/assets/")
    assert body["line_art_url"].startswith("/api/v1/curation/assets/")
    assert body["line_art_url"].endswith("/line-art")
    assert body["label_version"] == 0
    assert body["derivative_processing_state"] is None


def test_next_style_filter_only_returns_that_style(curation_client: TestClient) -> None:
    """Drain the queue for one style; every served candidate must match the filter."""
    seen_styles: set[str] = set()
    seen_assets: set[str] = set()
    while True:
        response = curation_client.get("/api/v1/curation/next", params={"style": "manga_anime"})
        if response.status_code == 404:
            break
        body = response.json()
        seen_styles.add(body["primary_style"])
        seen_assets.add(body["asset_id"])
        assert body["primary_style"] == "manga_anime"
        # Mark the candidate as rejected so the next ``/next`` advances;
        # the style filter would otherwise serve the same asset again.
        curation_client.post(
            "/api/v1/curation/labels",
            json=_label(body, decision="reject", quality=1),
        )
    assert seen_styles == {"manga_anime"}
    # The reject decisions must not have leaked into the other style buckets.
    body = curation_client.get("/api/v1/curation/progress").json()
    assert body["rejected"] == len(seen_assets)
    assert body["accepted"] == 0
    for style in ("western_ink", "realistic_academic", "cartoon", "gesture_sketch"):
        assert body["by_style"][style]["reviewed"] == 0
        assert body["by_style"][style]["rejected"] == 0


def test_next_scope_filter_only_returns_that_scope(curation_client: TestClient) -> None:
    """Drain the queue for one scope; the served candidates must respect the filter."""
    seen: set[tuple[str, ...]] = set()
    while True:
        response = curation_client.get("/api/v1/curation/next", params={"scope": "eye"})
        if response.status_code == 404:
            break
        body = response.json()
        scopes = (body["primary_scope"], *body["secondary_scopes"])
        seen.add(scopes)
        assert "eye" in scopes or scopes == ("unknown",)
        curation_client.post(
            "/api/v1/curation/labels",
            json=_label(body, decision="reject", quality=1),
        )
    assert seen  # we drained at least one candidate


def test_next_returns_404_when_queue_empty(curation_client: TestClient) -> None:
    # Drain the full queue by rejecting every served candidate. The queue
    # only advances when a label moves the asset out of ``unreviewed``.
    while True:
        response = curation_client.get("/api/v1/curation/next")
        if response.status_code == 404:
            break
        body = response.json()
        curation_client.post(
            "/api/v1/curation/labels",
            json=_label(body, decision="reject", quality=1),
        )
    response = curation_client.get("/api/v1/curation/next")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "queue_empty"


def test_next_404_without_gallery(tmp_path: Path) -> None:
    """Without a gallery manifest, the curation queue is meaningless."""
    with TestClient(
        create_app(make_settings(tmp_path, curation_mode=True, gallery_manifest=None))
    ) as client:
        response = client.get("/api/v1/curation/next")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "gallery_unavailable"


# ----------------------------------------------------------------- next (session queue)


def test_session_next_always_advances(curation_client: TestClient) -> None:
    """With a session id, repeated Next serves a fresh candidate every time.

    No label is written at all — the cursor alone advances the queue, so the
    reviewer can page through candidates before deciding anything.
    """
    session = "sess-advances"
    served: list[str] = []
    for _ in range(6):
        response = curation_client.get("/api/v1/curation/next", params={"session_id": session})
        assert response.status_code == 200
        body = response.json()
        assert body["asset_id"] not in served
        served.append(body["asset_id"])
    assert len(served) == 6


def test_session_next_wraps_at_end_without_repeating_back_to_back(
    curation_client: TestClient,
) -> None:
    """Past the end of the stable order the queue wraps — but never serves the
    same asset twice in a row."""
    session = "sess-wrap"
    # The synthetic fixture has 24 assets; page through more than that so the
    # wrap-around branch is exercised.
    served: list[str] = []
    previous = None
    for _ in range(30):
        response = curation_client.get("/api/v1/curation/next", params={"session_id": session})
        assert response.status_code == 200
        asset_id = response.json()["asset_id"]
        assert asset_id != previous
        previous = asset_id
        served.append(asset_id)
    # The whole unreviewed pool was covered by the wrap.
    assert len(set(served)) == 24


def test_session_skip_excludes_and_advances(curation_client: TestClient) -> None:
    """Skip advances without a label, and the skipped asset never comes back."""
    session = "sess-skip"
    first = curation_client.get("/api/v1/curation/next", params={"session_id": session}).json()
    second = curation_client.get("/api/v1/curation/next", params={"session_id": session}).json()

    response = curation_client.post(
        "/api/v1/curation/queue/skip",
        json={"session_id": session, "asset_id": second["asset_id"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cursor_asset_id"] == second["asset_id"]
    assert body["excluded_asset_id"] == second["asset_id"]
    # Remaining counts the session's non-excluded eligible pool: the skip is
    # the only thing that removes eligibility (serving just orders it).
    assert body["remaining"] == 23

    # The next serve is neither of the two we have seen...
    third = curation_client.get("/api/v1/curation/next", params={"session_id": session}).json()
    assert third["asset_id"] not in {first["asset_id"], second["asset_id"]}

    # ...and the skipped asset never returns, even after a full wrap.
    served: set[str] = set()
    for _ in range(30):
        nxt = curation_client.get("/api/v1/curation/next", params={"session_id": session}).json()
        served.add(nxt["asset_id"])
    assert second["asset_id"] not in served
    assert len(served) == 23  # everyone except the skipped asset

    # Skipping is per-session: another session still sees the asset.
    other = curation_client.get("/api/v1/curation/next", params={"session_id": "sess-other"})
    pool: set[str] = {second["asset_id"]}
    for _ in range(30):
        response = other
        if response.status_code == 404:
            break
        pool.add(response.json()["asset_id"])
        response = curation_client.get("/api/v1/curation/next", params={"session_id": "sess-other"})
    assert second["asset_id"] in pool


def test_skip_unknown_asset_is_404(curation_client: TestClient) -> None:
    response = curation_client.post(
        "/api/v1/curation/queue/skip",
        json={"session_id": "sess", "asset_id": "ls_none_0000000000000000"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "asset_not_found"


def test_label_with_session_advances_cursor(curation_client: TestClient) -> None:
    """A label written with a session id moves that session's queue forward."""
    session = "sess-label"
    first = curation_client.get("/api/v1/curation/next", params={"session_id": session}).json()
    assert (
        curation_client.post(
            "/api/v1/curation/labels", json=_label(first, session_id=session)
        ).status_code
        == 201
    )
    nxt = curation_client.get("/api/v1/curation/next", params={"session_id": session}).json()
    assert nxt["asset_id"] != first["asset_id"]


# ----------------------------------------------------------------- by-id retrieval


def test_get_candidate_by_id_returns_any_state(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))
    second = curation_client.get("/api/v1/curation/next").json()
    curation_client.post(
        "/api/v1/curation/labels", json=_label(second, decision="reject", quality=1)
    )

    # Accepted, rejected, and unreviewed assets are all retrievable by id —
    # Previous re-fetches live metadata instead of trusting a stale copy.
    accepted = _candidate(curation_client, first["asset_id"])
    assert accepted["review_state"] == "accepted"
    assert accepted["label_version"] == 1
    rejected = _candidate(curation_client, second["asset_id"])
    assert rejected["review_state"] == "rejected"
    assert rejected["label_version"] == 1
    third = curation_client.get("/api/v1/curation/next").json()
    unreviewed = _candidate(curation_client, third["asset_id"])
    assert unreviewed["review_state"] == "unreviewed"
    assert unreviewed["label_version"] == 0

    response = curation_client.get("/api/v1/curation/candidates/ls_none_0000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "asset_not_found"


def test_get_candidate_by_id_reflects_latest_state(curation_client: TestClient) -> None:
    """The by-id read path is live: a later write is immediately visible."""
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))
    fresh = _candidate(curation_client, first["asset_id"])
    assert fresh["review_state"] == "accepted"
    assert fresh["label_version"] == 1
    # The reviewer flips the decision; the by-id view follows immediately.
    curation_client.post(
        "/api/v1/curation/labels",
        json=_label(fresh, decision="reject", quality=2),
    )
    flipped = _candidate(curation_client, first["asset_id"])
    assert flipped["review_state"] == "rejected"
    assert flipped["label_version"] == 2


# ----------------------------------------------------------------- labels


def test_label_keep_quality_one_stays_disabled(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post("/api/v1/curation/labels", json=_label(first, quality=1))
    assert response.status_code == 201
    body = response.json()
    assert body["review_state"] == "accepted"
    assert body["review_quality"] == 1
    assert body["enabled"] is False
    assert body["label_version"] == 1
    state = curation_client.app.state.linescout
    enabled = {asset.asset_id for asset in enabled_assets(state.connection)}
    assert first["asset_id"] not in enabled


def test_curation_preview_serves_unreviewed_assets(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    public = curation_client.get(f"/api/v1/assets/{first['asset_id']}/thumbnail")
    assert public.status_code == 404
    preview = curation_client.get(first["thumbnail_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/png")
    line = curation_client.get(first["line_art_url"])
    assert line.status_code == 200


def test_label_keep_dual_writes_and_enables_asset(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post("/api/v1/curation/labels", json=_label(first))
    assert response.status_code == 201
    body = response.json()
    assert body["asset_id"] == first["asset_id"]
    assert body["decision"] == "keep"
    assert body["review_state"] == "accepted"
    assert body["review_quality"] == 3
    assert body["enabled"] is True
    assert body["label_version"] == 1

    # The asset is no longer a candidate: it has been accepted.
    after = curation_client.get("/api/v1/curation/next").json()
    assert after["asset_id"] != first["asset_id"]


def test_label_reject_disables_asset_and_records(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(first, decision="reject", quality=1, note="blurry"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["review_state"] == "rejected"
    assert body["enabled"] is False
    assert body["review_quality"] == 1

    # The in-memory ``enabled_assets`` list should no longer contain the
    # rejected asset, which is what /search and /assets/* observe.
    state = curation_client.app.state.linescout
    enabled = {asset.asset_id for asset in enabled_assets(state.connection)}
    assert first["asset_id"] not in enabled


def test_label_quality_required_on_keep(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json={
            "asset_id": first["asset_id"],
            "expected_review_state": "unreviewed",
            "expected_label_version": 0,
            "decision": "keep",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_label_validates_quality_range(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(first, quality=5),
    )
    assert response.status_code == 422


def test_label_validates_duplicate_scopes(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(
            first,
            primary_scope="eye",
            secondary_scopes=["eye", "eye"],
        ),
    )
    assert response.status_code == 422


def test_label_keep_with_blockers_is_422(curation_client: TestClient) -> None:
    """Blockers force reject or quarantine; a keep carrying them is invalid."""
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(first, blockers=["anatomy"]),
    )
    assert response.status_code == 422


def test_label_keep_with_unknown_primary_scope_is_422(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(first, primary_scope="unknown"),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_label_records_human_sfw_decision(curation_client: TestClient) -> None:
    """A keep with sfw_safe=true records the human approval and can serve."""
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(first, sfw_safe=True),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["sfw_human_approved"] is True
    state = curation_client.app.state.linescout
    row = state.connection.execute(
        "SELECT sfw_human_safe, sfw_human_reviewer FROM assets WHERE asset_id = ?",
        (first["asset_id"],),
    ).fetchone()
    assert row["sfw_human_safe"] == 1
    assert row["sfw_human_reviewer"] == "local"


def test_label_keep_without_permission_stays_disabled_with_reasons(
    curation_client: TestClient,
) -> None:
    """A keep on an asset with unknown permission is accepted but not servable,
    and the response names exactly why (no silent permission grant)."""
    first = curation_client.get("/api/v1/curation/next").json()
    state = curation_client.app.state.linescout
    state.connection.execute(
        "UPDATE assets SET allowed_display = 0, allowed_training = 0, allowed_trace = 0,"
        " permission_basis = 'unknown' WHERE asset_id = ?",
        (first["asset_id"],),
    )
    response = curation_client.post("/api/v1/curation/labels", json=_label(first))
    assert response.status_code == 201
    body = response.json()
    assert body["review_state"] == "accepted"
    assert body["enabled"] is False
    assert "display_not_permitted" in body["serving_blockers"]


def test_label_conflict_when_review_state_changed(curation_client: TestClient) -> None:
    """Two curators, same candidate: the second write is rejected atomically.

    The first curator's decision wins; the second gets a structured 409 with
    reconciliation information — never a silent overwrite (lost update).
    """
    first = curation_client.get("/api/v1/curation/next").json()
    ok = curation_client.post("/api/v1/curation/labels", json=_label(first))
    assert ok.status_code == 201
    # Curator B still holds the original unreviewed candidate (version 0).
    response = curation_client.post("/api/v1/curation/labels", json=_label(first))
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "review_conflict"
    assert body["error"]["field"] == "expected_review_state"
    assert body["retryable"] is False
    # Reconciliation information: the live version, the winning decision,
    # and how to re-apply. No filesystem paths.
    details = body["error"]["details"]
    assert details["current_label_version"] == 1
    assert details["current_review_state"] == "accepted"
    assert details["latest_decision"] == "keep"
    assert details["asset_id"] == first["asset_id"]
    assert "candidates/" in details["reconcile"]

    # Only ONE label row exists: the losing write was rolled back entirely.
    state = curation_client.app.state.linescout
    rows = state.connection.execute(
        "SELECT COUNT(*) AS n FROM curation_labels WHERE asset_id = ?",
        (first["asset_id"],),
    ).fetchone()
    assert rows["n"] == 1

    # Curator B reconciles: reload by id, re-apply against the live version.
    fresh = _candidate(curation_client, first["asset_id"])
    again = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(fresh, decision="reject", quality=2),
    )
    assert again.status_code == 201
    assert again.json()["label_version"] == 2


def test_label_version_conflict_when_only_version_changed(curation_client: TestClient) -> None:
    """A curation write that keeps the review state (SFW adjudication) still
    invalidates a stale label by version alone."""
    first = curation_client.get("/api/v1/curation/next").json()
    _hold(_state(curation_client), first["asset_id"])
    # An adjudication moves quarantined -> unreviewed and bumps the version.
    ok = curation_client.post(
        f"/api/v1/curation/sfw/{first['asset_id']}/adjudication",
        json={"safe": True, "expected_label_version": first["label_version"]},
    )
    assert ok.status_code == 201
    assert ok.json()["label_version"] == 1

    # Curator B labels against the stale version 0 with a matching state.
    response = curation_client.post("/api/v1/curation/labels", json=_label(first))
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "label_version_conflict"
    assert body["error"]["field"] == "expected_label_version"
    details = body["error"]["details"]
    assert details["current_label_version"] == 1
    assert details["current_review_state"] == "unreviewed"
    assert details["latest_sfw_adjudication"]["safe"] is True


def test_label_unknown_asset_404(curation_client: TestClient) -> None:
    response = curation_client.post(
        "/api/v1/curation/labels", json=_label("ls_does_not_exist_0000000000000000")
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "asset_not_found"


def test_label_updates_assets_metadata(curation_client: TestClient) -> None:
    """Reviewer-supplied style / scopes / blockers must persist.

    ``blockers`` live on both the audit row and the asset (they are named,
    use-blocking defects); the response echoes them. Crop geometry is no
    longer part of a label — crops are immutable derivatives with their own
    lifecycle (see the crop tests below).
    """
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        "/api/v1/curation/labels",
        json=_label(
            first,
            decision="reject",
            quality=1,
            primary_style="cartoon",
            primary_scope="full_body",
            secondary_scopes=["face_head"],
            blockers=["anatomy", "extraction"],
            note="head is too small",
        ),
    )
    assert response.status_code == 201

    state = curation_client.app.state.linescout
    asset_row = state.connection.execute(
        "SELECT primary_style, primary_scope, secondary_scopes_json, blockers_json"
        " FROM assets WHERE asset_id = ?",
        (first["asset_id"],),
    ).fetchone()
    assert asset_row["primary_style"] == "cartoon"
    assert asset_row["primary_scope"] == "full_body"
    assert json.loads(asset_row["secondary_scopes_json"]) == ["face_head"]
    assert json.loads(asset_row["blockers_json"]) == ["anatomy", "extraction"]
    scope_rows = state.connection.execute(
        "SELECT scope FROM asset_scopes WHERE asset_id = ? ORDER BY scope",
        (first["asset_id"],),
    ).fetchall()
    assert [row["scope"] for row in scope_rows] == ["face_head", "full_body"]

    label_row = state.connection.execute(
        "SELECT blockers_json, note"
        " FROM curation_labels WHERE asset_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (first["asset_id"],),
    ).fetchone()
    assert json.loads(label_row["blockers_json"]) == ["anatomy", "extraction"]
    assert label_row["note"] == "head is too small"


def test_label_progress_reflects_writes(curation_client: TestClient) -> None:
    body0 = curation_client.get("/api/v1/curation/progress").json()
    assert body0["reviewed"] == 0
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))
    body1 = curation_client.get("/api/v1/curation/progress").json()
    assert body1["reviewed"] == 1
    assert body1["accepted"] == 1
    assert body1["remaining"] == 1999
    assert body1["by_style"][first["primary_style"]]["accepted"] == 1


# ----------------------------------------------------------------- quarantine + reveal


def test_quarantine_list_is_metadata_only(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    second = curation_client.get("/api/v1/curation/next").json()
    state = _state(curation_client)
    _hold(state, first["asset_id"])
    state.connection.execute(
        "UPDATE assets SET sfw_verdict = 'unsure', sfw_confidence = 0.4,"
        " sfw_method = 'source_rating' WHERE asset_id = ?",
        (second["asset_id"],),
    )
    response = curation_client.get("/api/v1/curation/quarantine")
    assert response.status_code == 200
    entries = {entry["asset_id"]: entry for entry in response.json()}
    assert first["asset_id"] in entries
    assert second["asset_id"] in entries
    for entry in entries.values():
        # Metadata only: no image URLs before a deliberate reveal.
        assert entry["thumbnail_url"] is None
        assert entry["line_art_url"] is None
        assert entry["revealed"] is False
    screened = entries[second["asset_id"]]
    assert screened["sfw_screening"]["verdict"] == "unsure"


def test_preview_held_requires_reveal(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    _hold(_state(curation_client), first["asset_id"])

    preview = curation_client.get(first["thumbnail_url"])
    assert preview.status_code == 403
    body = preview.json()
    assert body["error"]["code"] == "reveal_required"
    assert body["retryable"] is False
    assert body["error"]["details"]["asset_id"] == first["asset_id"]

    # The public asset route stays closed — there is no public bypass.
    public = curation_client.get(f"/api/v1/assets/{first['asset_id']}/thumbnail")
    assert public.status_code == 404

    # A deliberate reveal opens the curation preview only.
    reveal = curation_client.post(
        f"/api/v1/curation/quarantine/{first['asset_id']}/reveal",
        json={"reviewer": "curator-a"},
    )
    assert reveal.status_code == 200
    revealed = reveal.json()
    assert revealed["revealed"] is True
    assert revealed["reveal_expires_at"] is not None
    assert revealed["thumbnail_url"] == first["thumbnail_url"]

    assert curation_client.get(first["thumbnail_url"]).status_code == 200
    assert curation_client.get(first["line_art_url"]).status_code == 200
    # ...and the public route still refuses.
    assert curation_client.get(f"/api/v1/assets/{first['asset_id']}/thumbnail").status_code == 404


def test_reveal_expires(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    _hold(_state(curation_client), first["asset_id"])
    assert (
        curation_client.post(
            f"/api/v1/curation/quarantine/{first['asset_id']}/reveal", json={}
        ).status_code
        == 200
    )
    # Expire the grant directly; the preview must fail closed again.
    state = _state(curation_client)
    state.connection.execute(
        "UPDATE sfw_reveals SET expires_at = '2000-01-01T00:00:00Z' WHERE asset_id = ?",
        (first["asset_id"],),
    )
    response = curation_client.get(first["thumbnail_url"])
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "reveal_required"
    listing = curation_client.get("/api/v1/curation/quarantine").json()
    entry = [item for item in listing if item["asset_id"] == first["asset_id"]][0]
    assert entry["revealed"] is False


def test_reveal_unheld_asset_is_422(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    response = curation_client.post(
        f"/api/v1/curation/quarantine/{first['asset_id']}/reveal", json={}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "asset_not_quarantined"
    missing = curation_client.post(
        "/api/v1/curation/quarantine/ls_none_0000000000000000/reveal", json={}
    )
    assert missing.status_code == 404


# ----------------------------------------------------------------- SFW adjudication


def test_adjudicate_safe_returns_record_to_queue(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    _hold(_state(curation_client), first["asset_id"])

    response = curation_client.post(
        f"/api/v1/curation/sfw/{first['asset_id']}/adjudication",
        json={"safe": True, "expected_label_version": first["label_version"], "reviewer": "medic"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["safe"] is True
    assert body["review_state"] == "unreviewed"  # back on its merits
    assert body["sfw_human"]["safe"] is True
    assert body["sfw_human"]["reviewer"] == "medic"
    assert body["label_version"] == 1
    # Not enabled yet: the record still needs a human keep with quality.
    assert body["enabled"] is False

    # The record rejoins the review queue and can now be accepted normally.
    served = set()
    for _ in range(30):
        nxt = curation_client.get("/api/v1/curation/next")
        if nxt.status_code == 404:
            break
        served.add(nxt.json()["asset_id"])
        curation_client.post(
            "/api/v1/curation/labels", json=_label(nxt.json(), decision="reject", quality=1)
        )
    assert first["asset_id"] in served

    fresh = _candidate(curation_client, first["asset_id"])
    accept = curation_client.post("/api/v1/curation/labels", json=_label(fresh, sfw_safe=True))
    assert accept.status_code == 201
    assert accept.json()["enabled"] is True

    # The adjudication is durable audit history.
    state = _state(curation_client)
    rows = state.connection.execute(
        "SELECT safe, prior_review_state, reviewer FROM sfw_adjudications WHERE asset_id = ?",
        (first["asset_id"],),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["safe"] == 1
    assert rows[0]["prior_review_state"] == "quarantined"
    assert rows[0]["reviewer"] == "medic"


def test_adjudicate_unsafe_quarantines_and_blocks_serving(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    keep = curation_client.post("/api/v1/curation/labels", json=_label(first, sfw_safe=True))
    assert keep.status_code == 201
    assert keep.json()["enabled"] is True
    assert curation_client.get(f"/api/v1/assets/{first['asset_id']}/thumbnail").status_code == 200

    response = curation_client.post(
        f"/api/v1/curation/sfw/{first['asset_id']}/adjudication",
        json={"safe": False, "expected_label_version": keep.json()["label_version"]},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["safe"] is False
    assert body["review_state"] == "quarantined"
    assert body["enabled"] is False
    assert body["label_version"] == 2

    # Every serving surface is closed, public and curation preview alike.
    assert curation_client.get(f"/api/v1/assets/{first['asset_id']}/thumbnail").status_code == 404
    assert curation_client.get(first["thumbnail_url"]).status_code == 403

    # The quarantined record shows up in the backlog metadata list.
    listing = curation_client.get("/api/v1/curation/quarantine").json()
    assert first["asset_id"] in {entry["asset_id"] for entry in listing}


def test_adjudication_version_conflict(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    _hold(_state(curation_client), first["asset_id"])
    ok = curation_client.post(
        f"/api/v1/curation/sfw/{first['asset_id']}/adjudication",
        json={"safe": True, "expected_label_version": 0},
    )
    assert ok.status_code == 201
    # A second adjudication against the stale version is a structured 409.
    response = curation_client.post(
        f"/api/v1/curation/sfw/{first['asset_id']}/adjudication",
        json={"safe": False, "expected_label_version": 0},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "label_version_conflict"
    assert body["error"]["details"]["current_label_version"] == 1
    # Nothing was written by the losing call.
    state = _state(curation_client)
    rows = state.connection.execute(
        "SELECT COUNT(*) AS n FROM sfw_adjudications WHERE asset_id = ?",
        (first["asset_id"],),
    ).fetchone()
    assert rows["n"] == 1


def test_adjudicate_non_pending_asset_is_422(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    # A plain unreviewed asset with no prior SFW decision at all: the
    # dedicated adjudication endpoint is not the place to record one.
    _state(curation_client).connection.execute(
        "UPDATE assets SET sfw_human_safe = NULL WHERE asset_id = ?",
        (first["asset_id"],),
    )
    response = curation_client.post(
        f"/api/v1/curation/sfw/{first['asset_id']}/adjudication",
        json={"safe": True, "expected_label_version": 0},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "asset_not_sfw_pending"


# ----------------------------------------------------------------- crops


def _make_crop(client: TestClient, candidate: dict, crop: dict, **overrides: object) -> object:
    return client.post(
        f"/api/v1/curation/assets/{candidate['asset_id']}/crops",
        json=_crop_body(candidate, crop, **overrides),
    )


def test_crop_out_of_bounds_is_422(crop_client: TestClient) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    response = _make_crop(
        crop_client,
        parent,
        {"x": 200, "y": 0, "width": 100, "height": 64},  # 256-wide source
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "crop_out_of_bounds"
    assert response.json()["error"]["field"] == "crop"


def test_crop_too_small_is_422(crop_client: TestClient) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    response = _make_crop(crop_client, parent, {"x": 0, "y": 0, "width": 8, "height": 8})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "crop_too_small"


def test_crop_on_held_parent_is_422(crop_client: TestClient) -> None:
    """Held content cannot be laundered into a fresh reviewable asset."""
    parent = crop_client.get("/api/v1/curation/next").json()
    _hold(_state(crop_client), parent["asset_id"])
    response = _make_crop(crop_client, parent, {"x": 0, "y": 0, "width": 64, "height": 64})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "crop_source_restricted"


def test_crop_version_conflict_is_409(crop_client: TestClient) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    response = _make_crop(
        crop_client,
        parent,
        {"x": 0, "y": 0, "width": 64, "height": 64},
        expected_label_version=parent["label_version"] + 5,
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "label_version_conflict"
    # The failed write left no child files behind (the derivatives root may
    # exist, but it is empty).
    derivatives_dir = _gallery_dir(crop_client) / "derivatives"
    assert not derivatives_dir.exists() or not any(derivatives_dir.iterdir())


def test_crop_creates_pending_derivative(crop_client: TestClient) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    crop = {"x": 32, "y": 32, "width": 64, "height": 64}
    response = _make_crop(crop_client, parent, crop, reviewer="curator-a", note="hand detail")
    assert response.status_code == 201
    child = response.json()
    assert child["parent_asset_id"] == parent["asset_id"]
    assert child["processing_state"] == "pending"
    assert child["derivatives_current"] is False
    assert child["review_state"] == "unreviewed"
    assert child["label_version"] == 0
    assert child["width"] == 64 and child["height"] == 64

    # Fresh files exist under the gallery's derivatives/ root.
    child_dir = _gallery_dir(crop_client) / "derivatives" / child["asset_id"]
    for name in ("original.png", "line_art.png", "thumbnail.png"):
        assert (child_dir / name).is_file()
    # The child's files are fresh bytes, not references to the parent's.
    parent_art = _gallery_dir(crop_client) / "line_art" / f"{parent['asset_id']}.png"
    child_art = child_dir / "line_art.png"
    assert (
        hashlib.sha256(child_art.read_bytes()).hexdigest()
        != hashlib.sha256(parent_art.read_bytes()).hexdigest()
    )

    # The registry row is the durable record: pending, fresh checksums, and
    # the parent's frozen artifact generation.
    state = _state(crop_client)
    registry = state.connection.execute(
        "SELECT * FROM curation_derivatives WHERE asset_id = ?", (child["asset_id"],)
    ).fetchone()
    assert registry is not None
    assert registry["processing_state"] == "pending"
    assert registry["parent_asset_id"] == parent["asset_id"]
    assert json.loads(registry["crop_json"]) == crop
    assert registry["line_art_checksum"] == hashlib.sha256(child_art.read_bytes()).hexdigest()
    assert registry["created_by"] == "curator-a"
    assert registry["note"] == "hand detail"

    # The child is visible by id (stable retrieval) but NOT reviewable yet:
    # it is not in the queue and a label fails closed with its problems.
    by_id = _candidate(crop_client, child["asset_id"])
    assert by_id["derivative_processing_state"] == "pending"
    assert by_id["parent_asset_id"] == parent["asset_id"]
    label = crop_client.post("/api/v1/curation/labels", json=_label(by_id))
    assert label.status_code == 422
    assert label.json()["error"]["code"] == "derivative_stale"
    assert label.json()["error"]["details"]["problems"] == ["derivative_awaiting_processing"]

    session = "sess-crop"
    for _ in range(40):
        nxt = crop_client.get("/api/v1/curation/next", params={"session_id": session})
        if nxt.status_code == 404:
            break
        assert nxt.json()["asset_id"] != child["asset_id"]

    # The parent itself is untouched by the crop.
    parent_after = _candidate(crop_client, parent["asset_id"])
    assert parent_after["label_version"] == parent["label_version"]
    assert parent_after["review_state"] == parent["review_state"]


def test_crop_is_immutable_and_conflicts_on_repeat(crop_client: TestClient) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    crop = {"x": 0, "y": 0, "width": 96, "height": 96}
    first = _make_crop(crop_client, parent, crop)
    assert first.status_code == 201
    child_id = first.json()["asset_id"]

    # The same exact crop is the same deterministic asset: 409, not a dupe.
    second = _make_crop(crop_client, parent, crop)
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "derivative_exists"
    state = _state(crop_client)
    rows = state.connection.execute(
        "SELECT COUNT(*) AS n FROM curation_derivatives WHERE parent_asset_id = ?",
        (parent["asset_id"],),
    ).fetchone()
    assert rows["n"] == 1

    # A different crop of the same parent is a different asset.
    other = _make_crop(crop_client, parent, {"x": 64, "y": 64, "width": 64, "height": 64})
    assert other.status_code == 201
    assert other.json()["asset_id"] != child_id


def test_crop_parent_files_verified(crop_client: TestClient) -> None:
    """A crop is only ever cut from verified parent artifacts.

    Tamper with the parent's line art on disk: the crop must fail closed and
    leave nothing behind.
    """
    parent = crop_client.get("/api/v1/curation/next").json()
    parent_art = _gallery_dir(crop_client) / "line_art" / f"{parent['asset_id']}.png"
    parent_art.write_bytes(b"tampered")

    response = _make_crop(crop_client, parent, {"x": 0, "y": 0, "width": 64, "height": 64})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "parent_artifact_invalid"
    assert "derivative_checksum_mismatch:line_art" in body["error"]["details"]["problems"]
    # No child row, no child files.
    state = _state(crop_client)
    rows = state.connection.execute("SELECT COUNT(*) AS n FROM curation_derivatives").fetchone()
    assert rows["n"] == 0
    derivatives_dir = _gallery_dir(crop_client) / "derivatives"
    assert not derivatives_dir.exists() or not any(derivatives_dir.iterdir())


# ----------------------------------------------------------------- derivative processing


def _process(client: TestClient, asset_id: str) -> object:
    return client.post(f"/api/v1/curation/assets/{asset_id}/process")


def test_process_derivative_completes_and_child_needs_own_approval(
    crop_client: TestClient,
) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    crop = {"x": 32, "y": 32, "width": 96, "height": 96}
    child = _make_crop(crop_client, parent, crop).json()

    response = _process(crop_client, child["asset_id"])
    assert response.status_code == 200
    body = response.json()
    assert body["processing_state"] == "complete"
    assert body["attempts"] == 1
    assert body["derivatives_current"] is True
    assert body["embedding_status"] == "missing"
    assert body["label_version"] == 1
    # Measurements were rebuilt from the child's own bytes.
    measurements = body["measurements"]
    assert measurements["width"] == 96 and measurements["height"] == 96
    assert 0.0 <= measurements["ink_coverage"] <= 1.0
    assert measurements["phash"] != "0" * 16
    # The artifact stamp binds future embeddings to these exact bytes.
    state = _state(crop_client)
    registry = state.connection.execute(
        "SELECT * FROM curation_derivatives WHERE asset_id = ?", (child["asset_id"],)
    ).fetchone()
    assert body["artifact"]["processing_revision"] == registry["processing_revision"]
    assert body["artifact"]["line_art_checksum"] == registry["line_art_checksum"]
    assert registry["processing_state"] == "complete"
    assert json.loads(registry["measurements_json"])["phash"] == measurements["phash"]

    # Serving still requires the child's OWN approval: nothing about the
    # parent's review/SFW state was inherited.
    assert body["enabled"] is False
    candidate = _candidate(crop_client, child["asset_id"])
    assert candidate["review_state"] == "unreviewed"
    assert candidate["label_version"] == 1
    assert candidate["derivative_processing_state"] == "complete"
    assert candidate["sfw_human"] is None
    # The parent may already be accepted; the child still needs its own keep.
    accept_parent = crop_client.post("/api/v1/curation/labels", json=_label(parent, sfw_safe=True))
    assert accept_parent.status_code == 201
    assert accept_parent.json()["enabled"] is True

    # The processed child is now in the queue and can earn approval itself.
    session = "sess-child"
    served = None
    for _ in range(40):
        nxt = crop_client.get("/api/v1/curation/next", params={"session_id": session})
        assert nxt.status_code == 200
        if nxt.json()["asset_id"] == child["asset_id"]:
            served = nxt.json()
            break
        crop_client.post(
            "/api/v1/curation/labels", json=_label(nxt.json(), decision="reject", quality=1)
        )
    assert served is not None
    accept = crop_client.post(
        "/api/v1/curation/labels",
        json=_label(served, primary_scope="hand", sfw_safe=True),
    )
    assert accept.status_code == 201
    assert accept.json()["enabled"] is True
    enabled = {asset.asset_id for asset in enabled_assets(_state(crop_client).connection)}
    assert child["asset_id"] in enabled


def test_process_failure_is_recorded_and_retryable(crop_client: TestClient) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    child = _make_crop(crop_client, parent, {"x": 0, "y": 0, "width": 64, "height": 64}).json()

    # Corrupt the child's line art after creation: processing must fail
    # closed, record the failure, and invalidate the child.
    child_art = _gallery_dir(crop_client) / "derivatives" / child["asset_id"] / "line_art.png"
    original_bytes = child_art.read_bytes()
    child_art.write_bytes(b"not a png")

    response = _process(crop_client, child["asset_id"])
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "derivative_files_invalid"
    assert "derivative_checksum_mismatch:line_art" in body["error"]["details"]["problems"]

    state = _state(crop_client)
    registry = state.connection.execute(
        "SELECT processing_state, processing_attempts, processing_error"
        " FROM curation_derivatives WHERE asset_id = ?",
        (child["asset_id"],),
    ).fetchone()
    assert registry["processing_state"] == "failed"
    assert registry["processing_attempts"] == 1
    assert registry["processing_error"] is not None
    asset_row = state.connection.execute(
        "SELECT derivatives_current, derivative_problems_json, curation_label_version"
        " FROM assets WHERE asset_id = ?",
        (child["asset_id"],),
    ).fetchone()
    assert asset_row["derivatives_current"] == 0
    assert "derivative_awaiting_processing" in json.loads(asset_row["derivative_problems_json"])
    assert asset_row["curation_label_version"] == 1

    # Retry after repairing the file: processing completes on the second go.
    child_art.write_bytes(original_bytes)
    retry = _process(crop_client, child["asset_id"])
    assert retry.status_code == 200
    assert retry.json()["processing_state"] == "complete"
    assert retry.json()["attempts"] == 2
    assert retry.json()["derivatives_current"] is True
    refreshed = _candidate(crop_client, child["asset_id"])
    assert refreshed["derivative_processing_state"] == "complete"
    assert refreshed["label_version"] == 2


def test_process_not_a_derivative_is_422(crop_client: TestClient) -> None:
    parent = crop_client.get("/api/v1/curation/next").json()
    response = _process(crop_client, parent["asset_id"])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "not_a_derivative"
    missing = _process(crop_client, "ls_none_0000000000000000")
    assert missing.status_code == 404


# ----------------------------------------------------------------- rehydration


def _manifest_variant(crop_client: TestClient, mutate) -> Path:
    """Write a mutated copy of the gallery manifest; return its path."""
    gallery_dir = _gallery_dir(crop_client)
    data = json.loads((gallery_dir / "manifest.json").read_text(encoding="utf-8"))
    mutate(data)
    target = gallery_dir.parent / "manifest_variant.json"
    # The manifest's data root is its parent directory: keep file paths valid
    # by writing the variant INTO the gallery directory.
    target = gallery_dir / "manifest_variant.json"
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def test_gallery_rebuild_rehydrates_derivatives(crop_client: TestClient) -> None:
    """A dataset reload neither loses derivatives nor resurrects stale state.

    The rebuild drops the whole ``assets`` cache; the crop child must be
    re-materialized from the durable registry with the parent's *current*
    row supplying identity/permissions and the child's own audit trail
    supplying its review state.
    """
    parent = crop_client.get("/api/v1/curation/next").json()
    child = _make_crop(crop_client, parent, {"x": 16, "y": 16, "width": 80, "height": 80}).json()
    assert _process(crop_client, child["asset_id"]).status_code == 200
    candidate = _candidate(crop_client, child["asset_id"])
    accept = crop_client.post(
        "/api/v1/curation/labels",
        json=_label(candidate, primary_scope="hand", sfw_safe=True),
    )
    assert accept.status_code == 201
    assert accept.json()["enabled"] is True

    child_before = (
        _state(crop_client)
        .connection.execute("SELECT * FROM assets WHERE asset_id = ?", (child["asset_id"],))
        .fetchone()
    )
    child_art_before = (
        _gallery_dir(crop_client) / "derivatives" / child["asset_id"] / "line_art.png"
    ).read_bytes()

    # A new manifest version arrives (same content, new dataset_version) and
    # the gallery is rebuilt from it.
    variant = _manifest_variant(
        crop_client, lambda data: data.update(dataset_version="2026.09.09-rebuilt")
    )
    gallery = sync_gallery(_state(crop_client).connection, variant)
    assert gallery is not None

    child_after = (
        _state(crop_client)
        .connection.execute("SELECT * FROM assets WHERE asset_id = ?", (child["asset_id"],))
        .fetchone()
    )
    assert child_after is not None, "the derivative must survive the rebuild"
    # Own review state restored from the audit trail, not the dropped cache.
    assert child_after["review_state"] == "accepted"
    assert child_after["review_quality"] == 3
    assert child_after["sfw_human_safe"] == 1
    assert child_after["primary_scope"] == "hand"
    assert child_after["curation_label_version"] == 1
    # Files and frozen generation are the child's own: unchanged bytes.
    assert child_after["line_art_checksum"] == child_before["line_art_checksum"]
    assert (
        _gallery_dir(crop_client) / "derivatives" / child["asset_id"] / "line_art.png"
    ).read_bytes() == child_art_before
    assert child_after["derivatives_current"] == 1
    assert child_after["enabled"] == 1
    # Scope mirror re-synced from the audit-trail state.
    scopes = (
        _state(crop_client)
        .connection.execute(
            "SELECT scope FROM asset_scopes WHERE asset_id = ? ORDER BY scope",
            (child["asset_id"],),
        )
        .fetchall()
    )
    assert [row["scope"] for row in scopes] == ["hand"]


def test_rebuild_propagates_revoked_parent_permissions(crop_client: TestClient) -> None:
    """A permission revoked in the new manifest reaches the crop child too —
    re-hydration must never resurrect stale grants."""
    parent = crop_client.get("/api/v1/curation/next").json()
    child = _make_crop(crop_client, parent, {"x": 0, "y": 0, "width": 64, "height": 64}).json()
    assert _process(crop_client, child["asset_id"]).status_code == 200
    candidate = _candidate(crop_client, child["asset_id"])
    accept = crop_client.post(
        "/api/v1/curation/labels",
        json=_label(candidate, primary_scope="eye", sfw_safe=True),
    )
    assert accept.json()["enabled"] is True

    def revoke(data: dict) -> None:
        data["dataset_version"] = "2026.09.10-revoked"
        for record in data["records"]:
            if record["asset_id"] == parent["asset_id"]:
                record["allowed_uses"]["display"] = False

    variant = _manifest_variant(crop_client, revoke)
    sync_gallery(_state(crop_client).connection, variant)

    state = _state(crop_client)
    child_row = state.connection.execute(
        "SELECT allowed_display, enabled, review_state FROM assets WHERE asset_id = ?",
        (child["asset_id"],),
    ).fetchone()
    assert child_row["allowed_display"] == 0
    assert child_row["enabled"] == 0  # the revoked grant propagated
    # The child's own review decision is preserved; only serving changed.
    assert child_row["review_state"] == "accepted"


def test_rebuild_marks_children_stale_on_generation_change(crop_client: TestClient) -> None:
    """A new artifact generation makes old crops stale until re-verified."""
    parent = crop_client.get("/api/v1/curation/next").json()
    child = _make_crop(crop_client, parent, {"x": 0, "y": 0, "width": 64, "height": 64}).json()
    assert _process(crop_client, child["asset_id"]).status_code == 200

    def bump_generation(data: dict) -> None:
        data["dataset_version"] = "2026.09.11-generation"
        data["artifact_contract"]["pipeline_version"] = "synthetic-99"
        # The manifest validator enforces gold eligibility against the
        # contract; a generation bump invalidates every gold member, so the
        # variant must ship without gold flags.
        for record in data["records"]:
            record["gold_member"] = False

    variant = _manifest_variant(crop_client, bump_generation)
    sync_gallery(_state(crop_client).connection, variant)

    state = _state(crop_client)
    row = state.connection.execute(
        "SELECT derivatives_current, derivative_problems_json, enabled"
        " FROM assets WHERE asset_id = ?",
        (child["asset_id"],),
    ).fetchone()
    problems = json.loads(row["derivative_problems_json"])
    assert row["derivatives_current"] == 0
    assert "derivative_stale:pipeline_version" in problems
    assert row["enabled"] == 0


def test_rebuild_without_parent_disables_child(crop_client: TestClient) -> None:
    """A child whose parent left the manifest survives, but inert."""
    parent = crop_client.get("/api/v1/curation/next").json()
    child = _make_crop(crop_client, parent, {"x": 0, "y": 0, "width": 64, "height": 64}).json()
    assert _process(crop_client, child["asset_id"]).status_code == 200

    def drop_parent(data: dict) -> None:
        data["dataset_version"] = "2026.09.12-orphan"
        data["records"] = [
            record for record in data["records"] if record["asset_id"] != parent["asset_id"]
        ]

    variant = _manifest_variant(crop_client, drop_parent)
    sync_gallery(_state(crop_client).connection, variant)

    state = _state(crop_client)
    row = state.connection.execute(
        "SELECT enabled, derivative_problems_json, allowed_display FROM assets WHERE asset_id = ?",
        (child["asset_id"],),
    ).fetchone()
    assert row is not None
    assert "derivative_parent_missing" in json.loads(row["derivative_problems_json"])
    assert row["allowed_display"] == 0
    assert row["enabled"] == 0


# ----------------------------------------------------------------- snapshots


def test_snapshot_writes_json_file_and_marks_labels(
    curation_client: TestClient, tmp_path: Path
) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))
    response = curation_client.post("/api/v1/curation/snapshots")
    assert response.status_code == 201
    body = response.json()
    assert body["snapshot_id"].startswith("curation_")
    assert body["label_count"] == 1
    assert not Path(body["path"]).is_absolute()
    assert body["path"].startswith("snapshots/")
    target = _data_dir(curation_client) / body["path"]
    assert target.is_file()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["snapshot_id"] == body["snapshot_id"]
    assert payload["label_count"] == 1
    assert payload["labels"][0]["asset_id"] == first["asset_id"]
    # The published bytes' hash is recorded and matches the file exactly.
    assert body["content_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    # Snapshot id is bound back onto the label row in the audit table.
    state = curation_client.app.state.linescout
    row = state.connection.execute(
        "SELECT snapshot_id FROM curation_labels WHERE asset_id = ?",
        (first["asset_id"],),
    ).fetchone()
    assert row["snapshot_id"] == body["snapshot_id"]


def test_snapshot_includes_keep_and_reject_labels(curation_client: TestClient) -> None:
    keep_id = curation_client.get("/api/v1/curation/next").json()["asset_id"]
    curation_client.post("/api/v1/curation/labels", json=_label(keep_id))

    reject_candidate = curation_client.get("/api/v1/curation/next").json()
    curation_client.post(
        "/api/v1/curation/labels", json=_label(reject_candidate, decision="reject", quality=1)
    )

    body = curation_client.post("/api/v1/curation/snapshots").json()
    assert body["label_count"] == 2
    payload = json.loads((_data_dir(curation_client) / body["path"]).read_text(encoding="utf-8"))
    by_id = {label["asset_id"]: label["decision"] for label in payload["labels"]}
    assert by_id == {keep_id: "keep", reject_candidate["asset_id"]: "reject"}


def test_snapshot_id_has_subsecond_precision(curation_client: TestClient) -> None:
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))
    body = curation_client.post("/api/v1/curation/snapshots").json()
    # curation_YYYYMMDD_HHMMSS_ffffff — microseconds so two exports in the
    # same second cannot share a filename.
    stamp = body["snapshot_id"].removeprefix("curation_")
    date, time_of_day, micros, *rest = stamp.split("_")
    assert len(date) == 8 and date.isdigit()
    assert len(time_of_day) == 6 and time_of_day.isdigit()
    assert len(micros) == 6 and micros.isdigit()
    assert body["path"] == f"snapshots/{body['snapshot_id']}.json"
    assert not Path(body["path"]).is_absolute()


def test_snapshot_404_without_gallery(tmp_path: Path) -> None:
    with TestClient(
        create_app(make_settings(tmp_path, curation_mode=True, gallery_manifest=None))
    ) as client:
        response = client.post("/api/v1/curation/snapshots")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "gallery_unavailable"


def test_snapshot_breaks_down_by_style(curation_client: TestClient) -> None:
    # Label one asset per style, then export the snapshot.
    seen_styles: set[str] = set()
    safety = 50  # the synthetic fixture has 24 assets; this is well under the cap.
    while len(seen_styles) < 2 and safety > 0:
        candidate = curation_client.get("/api/v1/curation/next").json()
        seen_styles.add(candidate["primary_style"])
        curation_client.post("/api/v1/curation/labels", json=_label(candidate))
        safety -= 1
    assert len(seen_styles) == 2, f"expected two distinct styles, got {seen_styles}"

    body = curation_client.post("/api/v1/curation/snapshots").json()
    assert body["label_count"] == 2
    total = sum(body["style_breakdown"].values())
    assert total == 2
    for style in seen_styles:
        assert body["style_breakdown"][style] == 1


# ------------------------------------------------------- snapshot semantics


def test_snapshots_are_full_and_latest_per_asset_with_lineage(
    curation_client: TestClient,
) -> None:
    """Frozen v2 snapshot semantics.

    * The snapshot captures the *latest* label per asset, rejected included.
    * A re-decision replaces the asset's entry; the audit history keeps both.
    * Every export is a new immutable file chained via ``previous_snapshot_id``
      — never an incremental delta of what changed since the last one.
    """
    keep_id = curation_client.get("/api/v1/curation/next").json()["asset_id"]
    assert curation_client.post("/api/v1/curation/labels", json=_label(keep_id)).status_code == 201

    reject_candidate = curation_client.get("/api/v1/curation/next").json()
    reject_id = reject_candidate["asset_id"]
    assert (
        curation_client.post(
            "/api/v1/curation/labels",
            json=_label(reject_candidate, decision="reject", quality=1),
        ).status_code
        == 201
    )

    first = curation_client.post("/api/v1/curation/snapshots").json()
    assert first["previous_snapshot_id"] is None
    assert first["label_count"] == 2

    # The reviewer changes their mind on the rejected asset: keep it now.
    fresh = _candidate(curation_client, reject_id)
    assert (
        curation_client.post(
            "/api/v1/curation/labels",
            json=_label(fresh, expected_review_state="rejected"),
        ).status_code
        == 201
    )

    second = curation_client.post("/api/v1/curation/snapshots").json()
    assert second["snapshot_id"] != first["snapshot_id"]
    assert second["previous_snapshot_id"] == first["snapshot_id"]
    # Full snapshot: still two assets (not one new label), and the latest
    # decision per asset wins.
    assert second["label_count"] == 2
    payload = json.loads((_data_dir(curation_client) / second["path"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["previous_snapshot_id"] == first["snapshot_id"]
    by_id = {label["asset_id"]: label["decision"] for label in payload["labels"]}
    assert by_id == {keep_id: "keep", reject_id: "keep"}

    # The append-only audit history still holds every decision, including both
    # of the flipped asset's rows.
    state = curation_client.app.state.linescout
    history = state.connection.execute(
        "SELECT decision FROM curation_labels WHERE asset_id = ? ORDER BY id",
        (reject_id,),
    ).fetchall()
    assert [row["decision"] for row in history] == ["reject", "keep"]

    # The first snapshot file was never touched by the second export.
    first_payload = json.loads(
        (_data_dir(curation_client) / first["path"]).read_text(encoding="utf-8")
    )
    assert {label["asset_id"]: label["decision"] for label in first_payload["labels"]} == {
        keep_id: "keep",
        reject_id: "reject",
    }


def test_snapshot_links_exactly_its_labels(curation_client: TestClient) -> None:
    """Only the label rows actually exported are stamped — a superseded label
    keeps the snapshot that first captured it."""
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))
    one = curation_client.post("/api/v1/curation/snapshots").json()

    fresh = _candidate(curation_client, first["asset_id"])
    curation_client.post(
        "/api/v1/curation/labels", json=_label(fresh, decision="reject", quality=1)
    )
    second = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(second))
    two = curation_client.post("/api/v1/curation/snapshots").json()

    state = curation_client.app.state.linescout
    rows = state.connection.execute(
        "SELECT id, decision, snapshot_id FROM curation_labels WHERE asset_id = ? ORDER BY id",
        (first["asset_id"],),
    ).fetchall()
    # The first (superseded) label keeps the first snapshot; the latest label
    # is stamped with the second one — exactly the rows that file contains.
    assert rows[0]["snapshot_id"] == one["snapshot_id"]
    assert rows[1]["snapshot_id"] == two["snapshot_id"]
    other = state.connection.execute(
        "SELECT snapshot_id FROM curation_labels WHERE asset_id = ?",
        (second["asset_id"],),
    ).fetchone()
    assert other["snapshot_id"] == two["snapshot_id"]


def test_snapshot_reproducible_after_later_edits(curation_client: TestClient) -> None:
    """A published snapshot is immutable and verifiable forever after.

    Later curation edits produce a *new* file; the earlier bytes (and their
    recorded content hash) never change.
    """
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))
    one = curation_client.post("/api/v1/curation/snapshots").json()
    path_one = _data_dir(curation_client) / one["path"]
    bytes_one = path_one.read_bytes()
    assert one["content_sha256"] == hashlib.sha256(bytes_one).hexdigest()

    # Later edits: flip the decision and label another asset.
    fresh = _candidate(curation_client, first["asset_id"])
    curation_client.post(
        "/api/v1/curation/labels", json=_label(fresh, decision="reject", quality=1)
    )
    nxt = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(nxt))
    two = curation_client.post("/api/v1/curation/snapshots").json()

    # The first file is byte-for-byte untouched and still matches its hash.
    assert path_one.read_bytes() == bytes_one
    state = curation_client.app.state.linescout
    registry = {
        row["snapshot_id"]: row["content_sha256"]
        for row in state.connection.execute(
            "SELECT snapshot_id, content_sha256 FROM snapshots"
        ).fetchall()
    }
    assert registry[one["snapshot_id"]] == one["content_sha256"]
    # The second export is a distinct file with its own recorded hash.
    path_two = _data_dir(curation_client) / two["path"]
    assert path_two.read_bytes() != bytes_one
    assert registry[two["snapshot_id"]] == hashlib.sha256(path_two.read_bytes()).hexdigest()
    assert two["content_sha256"] != one["content_sha256"]


def test_concurrent_exports_never_overwrite(curation_client: TestClient, monkeypatch) -> None:
    """Two exports racing on the shared connection serialize: one wins, the
    other gets a retryable 503 — never a mangled or overwritten file."""
    first = curation_client.get("/api/v1/curation/next").json()
    curation_client.post("/api/v1/curation/labels", json=_label(first))

    original = curation_module._publish_snapshot_exclusively

    def slow_publish(directory, base_id, payload):
        import time

        time.sleep(0.3)  # widen the race window deterministically
        return original(directory, base_id, payload)

    monkeypatch.setattr(curation_module, "_publish_snapshot_exclusively", slow_publish)

    state = curation_client.app.state.linescout
    results: list[tuple[int, str]] = []
    lock = threading.Lock()

    def run_export() -> None:
        try:
            response = curation_module.export_snapshot(state)
        except Exception as error:  # ApiError carries the status
            with lock:
                results.append(
                    (getattr(error, "status_code", 500), getattr(error, "code", "error"))
                )
        else:
            with lock:
                results.append((201, response.snapshot_id))

    threads = [threading.Thread(target=run_export) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(results) == 2

    statuses = sorted(status for status, _ in results)
    assert statuses == [201, 503], f"expected one success and one retryable failure, got {results}"
    loser = [code for status, code in results if status == 503][0]
    assert loser == "write_in_progress"

    # Exactly one snapshot file was published; nothing was overwritten.
    snapshots_dir = _data_dir(curation_client) / "snapshots"
    files = list(snapshots_dir.glob("*.json"))
    assert len(files) == 1
    registry = curation_client.app.state.linescout.connection.execute(
        "SELECT COUNT(*) AS n FROM snapshots"
    ).fetchone()
    assert registry["n"] == 1
