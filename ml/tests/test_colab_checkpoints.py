"""Tests for the checkpoint lock: pinned revisions, digests, and failing closed.

No downloads. Where a fetch would happen the verifier is given a fake fetcher that
writes bytes a test chose, so what is under test is the *decision* — accept,
record, or refuse — rather than a network.

    uv run pytest tests/test_colab_checkpoints.py -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from linescout_ml import colab
from linescout_ml.colab import checkpoints as cp
from linescout_ml.manifest import CheckpointProvenance, Manifest, PipelineProvenance

SHA = "a" * 40
DIGEST = "b" * 64

#: The digests published by lllyasviel/Annotators, read from its LFS metadata.
#: Naming them here as well as in the lock means a quiet edit to either is caught.
ANNOTATORS = {
    "netG.pth": (
        "ccabdcc3f5cf3c07cf65d58776acb21df7dfda825cdc70c9766a93fd62bfc488",
        217_631_959,
    ),
    "sk_model.pth": (
        "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
        17_173_511,
    ),
    "sk_model2.pth": (
        "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
        17_173_511,
    ),
}


def artifact(**overrides: Any) -> cp.CheckpointArtifact:
    payload: dict[str, Any] = {
        "id": "weights/net.pth",
        "group": "weights",
        "source": "huggingface",
        "repo": "someone/Annotators",
        "revision": SHA,
        "path": "net.pth",
        "sha256": DIGEST,
        "license": "MIT",
    }
    payload.update(overrides)
    return cp.CheckpointArtifact.model_validate(payload)


def lock_with(*artifacts: cp.CheckpointArtifact) -> cp.ModelLock:
    return cp.ModelLock.model_validate({"artifacts": [item.model_dump() for item in artifacts]})


def written(tmp_path: Path, payload: bytes, name: str = "net.pth") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def verifier_for(
    tmp_path: Path,
    *artifacts: cp.CheckpointArtifact,
    policy: str = "record",
    fetcher: Any = None,
) -> cp.CheckpointVerifier:
    lock = lock_with(*artifacts) if artifacts else cp.load_model_lock()
    return cp.CheckpointVerifier(
        lock,
        cache_dir=tmp_path / "cache",
        policy=policy,
        fetcher=fetcher,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------- the shipped lock


def test_every_stage_the_pipeline_can_run_has_a_pinned_group() -> None:
    """Not a nice-to-have: an unlisted group is a stage nobody pinned.

    The default embedders, every declared extractor, and the SFW gate are exactly
    the model loads a Colab run performs, so the lock has to answer for all of
    them. A group with no artifacts is not "nothing to pin" — it is a stage that
    will download whatever is current.
    """
    from linescout_ml.colab import extract, label

    lock = cp.load_model_lock()
    config = colab.PipelineConfig(
        dataset_version="2026.09.08-test",
        output_root=Path("/tmp/linescout-checkpoints"),
        sources=[colab.preset_source("synthetic", root=Path("/tmp/src"), license_id="cc0")],
    )
    groups = {
        *config.embedders,
        *label.NSFW_GROUP.split(),
        *(spec.checkpoint_group for spec in extract.EXTRACTOR_SPECS.values()),
    }
    for group in sorted(groups):
        assert lock.for_group(group), f"{group} has no checkpoint entry"
        for item in lock.for_group(group):
            assert item.repo or item.url, f"{item.id} names neither a repo nor a URL"


def test_the_annotator_digests_are_the_published_ones() -> None:
    """These three files decide every line drawing the pipeline produces."""
    lock = cp.load_model_lock()
    by_name = {item.filename: item for item in lock.for_group("informative_drawings")}
    by_name |= {item.filename: item for item in lock.for_group("anime2sketch")}

    for filename, (digest, size) in ANNOTATORS.items():
        entry = by_name[filename]
        assert entry.sha256 == digest, filename
        assert entry.size_bytes == size, filename
        assert entry.repo == "lllyasviel/Annotators"
        assert entry.revision and len(entry.revision) == 40


def test_a_pinned_revision_is_a_commit_for_hub_artifacts() -> None:
    """Hub and git artifacts name a commit; a `url` artifact may name a tag."""
    with pytest.raises(ValueError, match="not a 40-hex commit"):
        artifact(revision="main")
    with pytest.raises(ValueError, match="not a 40-hex commit"):
        artifact(source="git", repo="someone/Annotators", revision="v1.2")
    assert artifact(source="url", url="https://example.com/x.pth", revision="v0.1.0").revision


def test_the_lock_still_describes_itself_in_english() -> None:
    payload = json.loads(cp.package_lock_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["note"], "the lock has no explanation of its own conventions"
    rows = cp.summarize_lock(cp.load_model_lock())
    assert len(rows) == len(payload["artifacts"])
    assert {"group", "artifact", "revision", "sha256", "license"} <= set(rows[0])
    assert any(row["sha256"] == "unpinned" for row in rows), (
        "the lock claims a digest for a file no publisher hashes"
    )


def test_an_artifact_must_be_able_to_be_verified_at_all() -> None:
    """No digest, no size, and no explanation is not a pin — the loader refuses it.

    The rule binds the stages a default run loads. A repository *revision* is the
    pin for cloned code, so `git` artifacts carry a commit instead of a byte
    count; an opt-in model whose publisher exposes no size (DINOv2 ViT-B/14
    registers) is allowed in only if it *says* it is unpinned, so the report and
    `checkpoints show` both tell the truth about it.
    """
    lock = cp.load_model_lock()
    default_groups = {
        "anime2sketch",
        "informative_drawings",
        "mobileclip2_s2",
        "dinov2_vits14",
        "opennsfw2",
    }
    for item in lock.artifacts:
        if item.source == "git":
            assert item.revision and len(item.revision) == 40, item.id
        elif item.group in default_groups:
            assert item.sha256 or item.size_bytes, f"{item.id} cannot fail"
        else:
            assert item.note, f"{item.id} is unpinned and does not say so"

    broken = {
        "schema_version": 1,
        "artifacts": [
            {
                "id": "x/y.pth",
                "group": "x",
                "source": "url",
                "url": "https://e.com/y.pth",
                "path": "y.pth",
            }
        ],
    }
    with pytest.raises(ValueError, match="no note saying why"):
        cp.ModelLock.model_validate(broken)

    explained = json.loads(json.dumps(broken))
    explained["artifacts"][0]["note"] = "the publisher serves no size we can trust"
    assert cp.ModelLock.model_validate(explained).artifacts[0].pinned is False


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        lock_with(artifact(), artifact())


def test_the_lock_digest_names_the_file_that_produced_it(tmp_path: Path) -> None:
    lock = tmp_path / "models.lock.json"
    lock.write_text(
        '{"schema_version": 1, "artifacts": [{"id": "a/b", "group": "a",'
        ' "source": "url", "url": "https://e.com/b", "size_bytes": 3}]}\n'
    )
    digest = cp.lock_digest(lock)
    assert digest["sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert digest["path"].endswith("models.lock.json")
    assert cp.lock_digest(tmp_path / "absent.json")["sha256"] is None


# --------------------------------------------------------------------- verification


def test_a_matching_file_is_verified_and_reported_that_way(tmp_path: Path) -> None:
    body = b"weights"
    entry = artifact(sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
    verifier = verifier_for(tmp_path, entry)

    check = verifier.verify(entry, written(tmp_path, body))
    assert check.status == "verified" and check.pinned
    assert check.sha256 == entry.sha256 and check.size_bytes == len(body)
    assert verifier.verifications and "net.pth" in str(verifier.verifications[0].path)


@pytest.mark.parametrize("policy", ["strict", "record"])
def test_a_replaced_weight_file_stops_the_run(tmp_path: Path, policy: str) -> None:
    """The bytes on disk are not the reviewed ones — so nothing loads them."""
    entry = artifact(sha256=DIGEST)
    verifier = verifier_for(tmp_path, entry, policy=policy)

    with pytest.raises(cp.CheckpointMismatchError) as raised:
        verifier.verify(entry, written(tmp_path, b"different weights"))
    message = str(raised.value)
    assert "SHA-256 mismatch" in message
    assert DIGEST in message, "the refusal must name what was expected"
    assert "models.lock.json" in message, "and where to change it if the pin is wrong"


def test_a_truncated_download_is_caught_by_size_before_hashing(tmp_path: Path) -> None:
    body = b"x" * 100
    entry = artifact(sha256=hashlib.sha256(body).hexdigest(), size_bytes=1_000_000)
    verifier = verifier_for(tmp_path, entry)
    with pytest.raises(cp.CheckpointMismatchError, match="truncated or replaced"):
        verifier.verify(entry, written(tmp_path, body))


def test_a_missing_file_is_named_not_downloaded_under_verify(tmp_path: Path) -> None:
    entry = artifact()
    verifier = verifier_for(tmp_path, entry)
    with pytest.raises(cp.CheckpointError, match="found nothing"):
        verifier.verify(entry, tmp_path / "not-here.pth")


def test_policy_off_records_instead_of_refusing(tmp_path: Path) -> None:
    body = b"whatever the cache holds"
    entry = artifact(sha256=DIGEST)
    verifier = verifier_for(tmp_path, entry, policy="off")
    check = verifier.verify(entry, written(tmp_path, body))
    assert check.status == "unchecked" and check.pinned is True
    assert len(verifier.verifications) == 1


def test_strict_refuses_a_weight_no_publisher_hashed(tmp_path: Path) -> None:
    """`strict` means what it says: no digest, no run."""
    body = b"open_nsfw_weights"
    entry = artifact(
        id="opennsfw2/open_nsfw_weights.h5",
        group="opennsfw2",
        sha256=None,
        note="no digest published",  # allowed to exist, not allowed to run under strict
    )
    verifier = verifier_for(tmp_path, entry, policy="strict")
    with pytest.raises(cp.CheckpointUnpinnedError, match="requires a pinned sha256"):
        verifier.verify(entry, written(tmp_path, body, name="open_nsfw_weights.h5"))


def test_recorded_digests_are_enforced_on_the_next_run(tmp_path: Path) -> None:
    """Record-once, enforce-forever — not trust-on-first-use.

    A publisher that ships no hash (Meta's DINOv2 file, opennsfw2's weights) is
    the case the ledger exists for: the first run records what it saw, every later
    run verifies against it, and the only way to accept new bytes is to delete the
    record or pin the digest in a reviewed commit.
    """
    body = b"meta's weights, hashed by us"
    entry = artifact(sha256=None, size_bytes=len(body))
    first = verifier_for(tmp_path, entry, policy="record")
    check = first.verify(entry, written(tmp_path, body))
    assert check.status == "recorded"

    ledger = tmp_path / "cache" / "recorded.json"
    assert ledger.is_file()
    recorded = json.loads(ledger.read_text())
    digest = hashlib.sha256(body).hexdigest()
    assert recorded["artifacts"][entry.id]["sha256"] == digest
    assert recorded["artifacts"][entry.id]["locator"] == entry.locator()

    # What the ledger recorded is exactly what a reviewer would then pin, in a
    # reviewed commit: the same bytes read back as `verified`, not `recorded`.
    pinned = artifact(sha256=digest, size_bytes=len(body))
    assert first.verify(pinned, tmp_path / "net.pth").status == "verified"

    # Same cache, tampered file of the *same length*: the recorded digest, not the
    # byte count, is what catches a substituted weight file. Both the unpinned
    # entry (via the ledger) and the pinned one now refuse it.
    written(tmp_path, body[:-1] + b"!")
    second = verifier_for(tmp_path, entry, policy="record")
    with pytest.raises(cp.CheckpointMismatchError, match="SHA-256 mismatch"):
        second.verify(entry, tmp_path / "net.pth")
    with pytest.raises(cp.CheckpointMismatchError, match="SHA-256 mismatch"):
        second.verify(pinned, tmp_path / "net.pth")


# --------------------------------------------------------------------- fetch + staging


def test_a_fetch_is_verified_before_the_file_is_usable(tmp_path: Path) -> None:
    body = b"pinned bytes"
    entry = artifact(sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
    calls: list[Path] = []

    def fetch(_artifact: cp.CheckpointArtifact, destination: Path) -> None:
        calls.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)

    verifier = verifier_for(tmp_path, entry, fetcher=fetch)
    path = verifier.file(entry.id)

    assert path.is_file() and path.read_bytes() == body
    assert calls == [path], "the cache was populated exactly once"
    assert verifier.file(entry.id) == path and len(calls) == 1, (
        "a verified cache hit must not refetch"
    )
    assert verifier.verifications[0].status == "verified"


def test_a_download_that_is_not_the_pinned_bytes_stops_the_run(tmp_path: Path) -> None:
    """The verify-after-fetch step, and the evidence it leaves behind.

    Deleting bytes nobody asked for is not this pipeline's job: the file stays so
    the next person can compare it with what the lock expects.
    """
    entry = artifact(sha256=DIGEST)

    def fetch(_artifact: cp.CheckpointArtifact, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"wrong")

    verifier = verifier_for(tmp_path, entry, fetcher=fetch)
    with pytest.raises(cp.CheckpointMismatchError, match="SHA-256 mismatch"):
        verifier.file(entry.id)
    cached = tmp_path / "cache" / "files" / entry.id
    assert cached.read_bytes() == b"wrong", "the refusal has to leave evidence to compare"


def test_a_stale_cached_file_is_never_silently_replaced(tmp_path: Path) -> None:
    entry = artifact(sha256=hashlib.sha256(b"new").hexdigest())
    destination = tmp_path / "cache" / "files" / entry.id
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale")

    def fetch(_artifact: cp.CheckpointArtifact, path: Path) -> None:  # pragma: no cover
        raise AssertionError("a mismatching cache entry must stop the run, not trigger a refetch")

    verifier = verifier_for(tmp_path, entry, fetcher=fetch)
    with pytest.raises(cp.CheckpointMismatchError, match="delete it and re-run"):
        verifier.file(entry.id)


def test_a_group_directory_holds_every_verified_file(tmp_path: Path) -> None:
    """What a loader is handed: a folder of verified bytes, no repo id in sight."""
    first, second = b"one", b"two"
    entries = [
        artifact(
            id="annotators/netG.pth",
            group="annotators",
            path="netG.pth",
            sha256=hashlib.sha256(first).hexdigest(),
            size_bytes=len(first),
        ),
        artifact(
            id="annotators/sk_model.pth",
            group="annotators",
            path="sk_model.pth",
            sha256=hashlib.sha256(second).hexdigest(),
            size_bytes=len(second),
        ),
    ]
    for entry, body in zip(entries, (first, second), strict=True):
        destination = tmp_path / "cache" / "files" / entry.id
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)

    verifier = verifier_for(tmp_path, *entries)
    folder = verifier.directory("annotators")
    assert sorted(path.name for path in folder.iterdir()) == ["netG.pth", "sk_model.pth"]
    assert {entry.id: entry.filename for entry in verifier.artifacts("annotators")} == {
        "annotators/netG.pth": "netG.pth",
        "annotators/sk_model.pth": "sk_model.pth",
    }

    # A tampered staged copy of the same length is caught too: staging verifies,
    # it does not trust the folder it was given.
    (folder / "netG.pth").write_bytes(b"on!")
    with pytest.raises(cp.CheckpointMismatchError, match="SHA-256 mismatch"):
        verifier_for(tmp_path, *entries).directory("annotators")


def test_a_group_the_lock_does_not_know_is_an_error(tmp_path: Path) -> None:
    verifier = verifier_for(tmp_path, artifact())
    assert verifier.knows("weights") and not verifier.knows("nothing")
    with pytest.raises(cp.CheckpointError, match="no artifacts for group"):
        verifier.directory("nothing")
    with pytest.raises(cp.CheckpointError, match="unknown checkpoint artifact"):
        verifier.artifact("nope/nope.pth")


def test_code_only_artifacts_are_pinned_by_revision(tmp_path: Path) -> None:
    """torch.hub clones code; the commit is the pin, so there is nothing to hash."""
    entry = artifact(source="git", id="dinov2_vits14/hubconf.py", path=None, sha256=None)
    verifier = verifier_for(tmp_path, entry)
    assert verifier.hub_repo_ref(entry.group) == f"{entry.repo}:{SHA}"
    assert verifier.revisions(entry.group) == {entry.repo: SHA}
    path = verifier.file(entry.id)
    assert path == tmp_path / "cache" / "git" / entry.id
    assert verifier.verifications[0].status == "revision"


def test_dinov2_and_opennsfw2_resolve_their_pinned_revision() -> None:
    """The loaders that need a *repo ref* rather than a path, and the file names.

    Read from the lock only: touching the network to prove a pin is not a test.
    """
    lock = cp.load_model_lock()
    verifier = cp.CheckpointVerifier(lock, cache_dir=Path("/tmp/linescout-cache"))
    assert verifier.hub_repo_ref("dinov2_vits14") == (
        "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    )
    gate = lock.by_id()["opennsfw2/open_nsfw_weights.h5"]
    assert gate.revision == "v0.1.0" and gate.size_bytes == 24_221_200
    assert gate.sha256 is None, (
        "opennsfw2 publishes no digest; pin one here only once a real one exists"
    )
    assert verifier.weights_file.__doc__, "weights_file must keep saying what it returns"


def test_ensure_at_checks_a_file_a_library_chose(tmp_path: Path) -> None:
    """opennsfw2 insists on its own default path; verify there, do not move it."""
    body = b"opennsfw2 weights"
    entry = artifact(
        id="opennsfw2/open_nsfw_weights.h5",
        group="opennsfw2",
        path="open_nsfw_weights.h5",
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
    )
    elsewhere = tmp_path / ".opennsfw2" / "weights" / "open_nsfw_weights.h5"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(body)

    verifier = cp.CheckpointVerifier(lock_with(entry), cache_dir=tmp_path / "cache")
    assert verifier.ensure_at(entry.id, elsewhere).status == "verified"

    elsewhere.write_bytes(body + b" and more")
    with pytest.raises(cp.CheckpointMismatchError):
        verifier.ensure_at(entry.id, elsewhere)


# ----------------------------------------------------------------------- wiring


def test_the_policy_comes_from_the_run_config(tmp_path: Path) -> None:
    lock = tmp_path / "models.lock.json"
    lock.write_text(json.dumps({"artifacts": [artifact(group="custom").model_dump()]}))
    config = SimpleNamespace(
        checkpoint_policy="strict", model_lock_path=lock, checkpoint_cache_dir=tmp_path / "c"
    )

    verifier = cp.CheckpointVerifier.for_config(config)
    assert verifier.policy == "strict"
    assert verifier.cache_dir == tmp_path / "c"
    assert [item.group for item in verifier.lock.artifacts] == ["custom"]

    off = cp.CheckpointVerifier.for_config(
        SimpleNamespace(checkpoint_policy="off", model_lock_path=lock, checkpoint_cache_dir=None)
    )
    assert off.policy == "off"
    assert off.file("weights/net.pth") == off.cache_dir / "files" / "weights/net.pth"

    with pytest.raises(cp.CheckpointError, match="unknown checkpoint policy"):
        cp.CheckpointVerifier(lock_with(artifact()), policy="whatever")  # type: ignore[arg-type]


def test_the_default_policy_records_rather_than_refuses() -> None:
    """The default has to work on a first run *and* be honest about what it saw."""
    assert colab.PipelineConfig.model_fields["checkpoint_policy"].default == "record"
    assert cp.CheckpointVerifier.disabled().policy == "off"


def test_every_loader_asked_for_the_right_group() -> None:
    """A typo in a group name would silently skip verification for that model."""
    from linescout_ml.colab import extract, label, models

    lock = cp.load_model_lock()
    for key, spec in extract.EXTRACTOR_SPECS.items():
        if spec.checkpoint_group:
            assert lock.for_group(spec.checkpoint_group), f"extractor {key}"
    for key, card in models.MODEL_CARDS.items():
        if card.checkpoint_group:
            assert lock.for_group(card.checkpoint_group) or key in cp.missing_groups(
                lock, [card.checkpoint_group]
            ), f"embedder {key}"
    assert label.NSFW_GROUP == "opennsfw2" and lock.for_group(label.NSFW_GROUP)


def test_an_opt_in_model_without_a_pin_is_refused_rather_than_fetched() -> None:
    """MobileCLIP2-S0 is selectable but not pinned, and that has to bite.

    Its weights have no published digest the project trusts, so the lock names no
    files for it. A loader that was handed `""` as its checkpoint group would
    quietly download whatever the upstream tag points at today; the card names a
    group instead, so the run stops with a sentence about it.
    """
    from linescout_ml.colab import models

    lock = cp.load_model_lock()
    card = models.MODEL_CARDS["mobileclip2_s0"]
    assert card.checkpoint_group == "mobileclip2_s0"
    assert cp.missing_groups(lock, [card.checkpoint_group]) == ["mobileclip2_s0"]
    verifier = cp.CheckpointVerifier(lock, cache_dir=Path("/tmp/linescout-s0"))
    with pytest.raises(cp.CheckpointError, match="no artifacts for group"):
        verifier.directory("mobileclip2_s0")


def test_the_manifest_carries_the_verification_outcome(tmp_path: Path) -> None:
    """The ledger in `run_report.json` has to reach the dataset too.

    A provenance block that could not change a content hash would be decoration:
    two datasets built from different weights must not hash alike.
    """
    body = b"weights"
    entry = artifact(sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
    verifier = verifier_for(tmp_path, entry)
    verifier.verify(entry, written(tmp_path, body))
    payload = verifier.as_dict()

    assert payload["policy"] == "record"
    assert payload["lock"]["sha256"]
    assert payload["artifacts"][0]["id"] == entry.id

    provenance = PipelineProvenance(
        pipeline_version="colab-m2-1",
        source_revision=SHA,
        checkpoint_policy="record",
        # `path` is a local absolute path: right for the run report, never for a
        # manifest that gets published. The runner strips it the same way.
        checkpoints=[
            CheckpointProvenance.model_validate(
                {key: value for key, value in payload["artifacts"][0].items() if key != "path"}
            )
        ],
    )
    first = Manifest(dataset_version="2026.09.08-test", provenance=provenance, records=[])
    second = first.model_copy(
        update={
            "provenance": provenance.model_copy(
                update={"checkpoints": [*provenance.checkpoints[:-1]]}
            )
        }
    )
    assert first.provenance is not None
    assert first.provenance.checkpoints[0].sha256 == entry.sha256
    assert first.content_hash() != second.content_hash(), (
        "a run with different verified weights must not look identical"
    )
