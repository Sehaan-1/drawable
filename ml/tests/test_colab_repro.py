"""Tests for the parts of reproducibility a lockfile cannot deliver.

No network, no GPU: the "remote" is a git repository in ``tmp_path`` and the
environment spec is text a test wrote (or the committed one, read read-only).
What is under test is the surrounding contract the Colab notebook depends on —
which commit the code came from and whether that was *verified*, which packages a
resolved run needs, and what the runtime actually had installed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from linescout_ml import colab
from linescout_ml.colab import repro

REPO_ROOT = repro.repo_root()


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def make_origin(root: Path) -> tuple[Path, str]:
    """A stand-in for the pinned repository: a checkout with an ``ml/`` tree."""
    origin = root / "origin"
    (origin / "ml" / "linescout_ml" / "colab").mkdir(parents=True)
    (origin / "ml" / "pyproject.toml").write_text("[project]\nname = 'linescout-ml'\n")
    (origin / "ml" / "linescout_ml" / "colab" / "marker.py").write_text("REV = 1\n")
    git(origin, "init", "--quiet", "--initial-branch=main", ".")
    git(origin, "config", "user.email", "test@example.com")
    git(origin, "config", "user.name", "Test")
    git(origin, "add", "-A")
    git(origin, "commit", "--quiet", "-m", "first")
    return origin, git(origin, "rev-parse", "HEAD")


def advance(origin: Path, revision: int) -> str:
    """One more commit on main, so "whatever main says today" means something."""
    (origin / "ml" / "linescout_ml" / "colab" / "marker.py").write_text(f"REV = {revision}\n")
    git(origin, "add", "-A")
    git(origin, "commit", "--quiet", "-m", f"revision {revision}")
    return git(origin, "rev-parse", "HEAD")


# ------------------------------------------------------------------ the pinned checkout


def test_a_clone_lands_on_the_pinned_commit_and_verifies_head(tmp_path: Path) -> None:
    origin, first = make_origin(tmp_path)
    advance(origin, 2)  # main has moved on; the pin must not follow it

    record = repro.ensure_checkout(tmp_path / "work", revision=first, url=str(origin), mirrors=())

    assert record.verified and record.usable and record.matches_pin
    assert record.revision == first
    assert record.action == "cloned"
    assert record.dirty is False and record.detached is True
    assert (tmp_path / "work/ml/linescout_ml/colab/marker.py").read_text() == "REV = 1\n"


def test_the_provenance_names_the_commit_the_run_used(tmp_path: Path) -> None:
    origin, pinned = make_origin(tmp_path)
    record = repro.ensure_checkout(tmp_path / "work", revision=pinned, url=str(origin), mirrors=())

    provenance = repro.checkout_provenance(record)
    assert provenance["revision"] == pinned
    assert provenance["requested_revision"] == pinned
    assert provenance["dirty"] is False
    assert provenance["verified"] is True
    assert provenance["action"] == "cloned"
    # A local path is not a GitHub remote, so there is no owner/name to claim.
    assert provenance["repo"] == repro.COLAB_REPO

    stamped = colab.PipelineConfig(
        dataset_version="2026.09.08-test",
        output_root=tmp_path / "gallery",
        sources=[colab.preset_source("synthetic", root=tmp_path / "src", license_id="cc0")],
        source_repo="someone/else",
        source_revision=pinned,
        source_dirty=False,
        source_action="cloned",
    )
    assert stamped.source_provenance() is not None
    assert stamped.source_provenance()["revision"] == pinned


def test_a_config_with_nothing_verified_is_distinguishable_from_a_clean_tree(
    tmp_path: Path,
) -> None:
    """`None` means "we did not look", which is not the same as "we looked"."""
    bare = colab.PipelineConfig(
        dataset_version="2026.09.08-test",
        output_root=tmp_path / "gallery",
        sources=[colab.preset_source("synthetic", root=tmp_path / "src", license_id="cc0")],
    )
    assert bare.source_provenance() is None

    partial = bare.model_copy(update={"source_dirty": False})
    assert partial.source_provenance() is not None
    assert partial.source_provenance()["revision"] is None


@pytest.mark.parametrize("revision", ["main", "HEAD", "v1.0.0", "abc1234", "", None, 7])
def test_only_a_full_commit_sha_counts_as_immutable(revision: object) -> None:
    """Branches and tags move, so they are refused before git is even asked."""
    with pytest.raises(repro.CheckoutError, match="immutable|40-hex"):
        repro.require_immutable_revision(revision)


def test_an_abbreviated_sha_is_refused_too(tmp_path: Path) -> None:
    origin, pinned = make_origin(tmp_path)
    with pytest.raises(repro.CheckoutError, match="40-hex"):
        repro.ensure_checkout(tmp_path / "work", revision=pinned[:10], url=str(origin), mirrors=())
    assert not (tmp_path / "work/.git").exists(), "the refusal still created a checkout"


def test_a_clean_checkout_behind_the_pin_is_fetched_onto_it(tmp_path: Path) -> None:
    origin, first = make_origin(tmp_path)
    work = tmp_path / "work"
    repro.ensure_checkout(work, revision=first, url=str(origin), mirrors=())
    assert (work / "ml/linescout_ml/colab/marker.py").read_text() == "REV = 1\n"

    second = advance(origin, 2)
    record = repro.ensure_checkout(work, revision=second, url=str(origin), mirrors=())

    assert record.action == "fetched" and record.verified
    assert record.revision == second
    assert (work / "ml/linescout_ml/colab/marker.py").read_text() == "REV = 2\n"


def test_an_existing_checkout_at_the_pin_is_reused_untouched(tmp_path: Path) -> None:
    origin, pinned = make_origin(tmp_path)
    work = tmp_path / "work"
    repro.ensure_checkout(work, revision=pinned, url=str(origin), mirrors=())
    (work / "notes.txt").write_text("scratch\n")  # untracked output, not a code edit

    record = repro.ensure_checkout(work, revision=pinned, url=str(origin), mirrors=())
    assert record.action == "reused" and record.verified
    assert record.untracked == 1 and record.dirty is True
    assert (work / "notes.txt").is_file()


def test_a_dirty_checkout_at_the_wrong_revision_is_refused_and_left_alone(
    tmp_path: Path,
) -> None:
    """The pipeline may not destroy work it cannot judge."""
    origin, first = make_origin(tmp_path)
    work = tmp_path / "work"
    repro.ensure_checkout(work, revision=first, url=str(origin), mirrors=())
    marker = work / "ml/linescout_ml/colab/marker.py"
    marker.write_text("REV = my uncommitted investigation\n")
    second = advance(origin, 2)

    with pytest.raises(repro.RevisionMismatchError, match="Refusing to move"):
        repro.ensure_checkout(work, revision=second, url=str(origin), mirrors=())

    assert marker.read_text() == "REV = my uncommitted investigation\n"
    assert git(work, "rev-parse", "HEAD") == first


def test_dirty_at_the_pin_is_recorded_and_can_optionally_be_refused(
    tmp_path: Path,
) -> None:
    origin, pinned = make_origin(tmp_path)
    work = tmp_path / "work"
    repro.ensure_checkout(work, revision=pinned, url=str(origin), mirrors=())
    (work / "ml/pyproject.toml").write_text("[project]\nname = 'edited'\n")

    record = repro.ensure_checkout(work, revision=pinned, url=str(origin), mirrors=())
    assert record.verified and record.dirty and record.action == "reused"
    assert any("local edits" in note for note in record.notes)
    assert "dirty" in json.dumps(record.as_dict())

    with pytest.raises(repro.CheckoutError, match="refuse a dirty tree"):
        repro.ensure_checkout(work, revision=pinned, url=str(origin), mirrors=(), allow_dirty=False)
    assert (work / "ml/pyproject.toml").read_text().endswith("edited'\n")


def test_a_mirror_cannot_smuggle_in_different_code(tmp_path: Path) -> None:
    """The mirror is tried only when the canonical URL fails, and HEAD still rules."""
    good, pinned = make_origin(tmp_path)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "README.md").write_text("not a repository\n")

    record = repro.ensure_checkout(
        tmp_path / "work",
        revision=pinned,
        url=str(tmp_path / "does-not-exist"),
        mirrors=(str(bad), str(good)),
    )
    assert record.verified and record.source_url == str(good)

    # An origin that cannot answer for the pin is refused, and says who it asked.
    with pytest.raises(repro.CheckoutError, match="could not obtain") as raised:
        repro.ensure_checkout(
            tmp_path / "other-work",
            revision=pinned,
            url=str(tmp_path / "no-such-repo"),
            mirrors=(str(tmp_path / "also-nothing"),),
        )
    assert "no-such-repo" in str(raised.value) and "also-nothing" in str(raised.value)
    assert not (tmp_path / "other-work/ml").exists(), "a refused clone left a half-tree behind"


def test_a_placeholder_pin_is_named_as_such_when_it_fails(tmp_path: Path) -> None:
    """The transitional state of a repository whose pin has not been bumped yet.

    A row of zeroes is a syntactically valid SHA that no repository contains, so
    the refusal has to say what it is instead of looking like a network fault.
    """
    origin, _ = make_origin(tmp_path)
    with pytest.raises(repro.CheckoutError, match="placeholder pin"):
        repro.ensure_checkout(
            tmp_path / "work",
            revision=repro.COLAB_PIN_PLACEHOLDER,
            url=str(origin),
            mirrors=(),
        )
    assert not repro.pin_is_finalised(repro.COLAB_PIN_PLACEHOLDER)
    assert repro.pin_is_finalised("a" * 40)


def test_a_non_empty_directory_that_is_not_a_repo_is_not_written_into(
    tmp_path: Path,
) -> None:
    origin, pinned = make_origin(tmp_path)
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "data.parquet").write_text("precious\n")

    with pytest.raises(repro.CheckoutError, match="refusing to clone into it"):
        repro.ensure_checkout(target, revision="a" * 40, url=str(origin), mirrors=())
    assert (target / "data.parquet").is_file()


def test_preferred_checkouts_are_verified_not_trusted(tmp_path: Path) -> None:
    origin, pinned = make_origin(tmp_path)
    drive = tmp_path / "drive-copy"
    repro.ensure_checkout(drive, revision=pinned, url=str(origin), mirrors=())
    (drive / "ml/linescout_ml/colab/marker.py").write_text("REV = 1\n")

    record = repro.find_or_checkout(
        revision=pinned, url=str(origin), mirrors=(), preferred=[tmp_path / "nope", drive]
    )
    assert Path(record.path) == drive and record.action == "reused"

    # Nothing usable: clone into the default target rather than giving up.
    fresh = repro.find_or_checkout(
        revision=pinned,
        url=str(origin),
        mirrors=(),
        preferred=[tmp_path / "nothing"],
        target=tmp_path / "cloned",
    )
    assert Path(fresh.path) == tmp_path / "cloned" and fresh.verified


# ------------------------------------------------------------------- environment spec


SPEC_TEXT = """\
# group: base
numpy==2.1.0
pillow==11.0.0
# group: extraction
controlnet-aux==0.0.10
# group: nsfw
opennsfw2==0.18.0
tensorflow-cpu==2.17.0
# group: runtime
torch==2.5.1
torchvision==0.20.1
"""


def test_the_spec_groups_decide_what_is_installable(tmp_path: Path) -> None:
    spec_file = tmp_path / "requirements-colab.txt"
    spec_file.write_text(SPEC_TEXT)
    spec = repro.load_requirement_spec(spec_file)
    assert spec is not None

    assert "opennsfw2" in spec.groups["nsfw"] and "tensorflow-cpu" in spec.groups["nsfw"]
    assert spec.runtime_only == ("torch", "torchvision"), "the runtime group is never installed"
    assert spec.pins["controlnet-aux"] == "0.0.10"
    assert spec.sha256 == hashlib.sha256(spec_file.read_bytes()).hexdigest()


def test_a_missing_spec_is_none_rather_than_an_empty_plan(tmp_path: Path) -> None:
    assert repro.load_requirement_spec(tmp_path / "absent.txt") is None
    assert repro.parse_requirements("numpy\n").pins == {}  # unpinned lines pin nothing


def test_the_committed_spec_agrees_with_the_committed_lockfile() -> None:
    """Two files, one answer — checked here as well as in CI's selfcheck."""
    spec = repro.load_requirement_spec()
    assert spec is not None
    locked = repro.lockfile_versions(REPO_ROOT / "ml/uv.lock")
    for name, version in spec.pins.items():
        assert name in locked, f"{name} is pinned for Colab but not in uv.lock"
        assert version in locked[name], f"{name}: spec {version}, lock {locked[name]}"


def test_the_runtime_group_is_the_only_place_torch_appears() -> None:
    spec = repro.load_requirement_spec()
    assert spec is not None
    installable = {
        name for group, names in spec.groups.items() if group != "runtime" for name in names
    }
    assert not installable & {"torch", "torchvision", "nvidia"}
    assert "torch" in spec.groups["runtime"]


# ------------------------------------------------------------------ what a run needs


def specs_for(*entries: dict[str, Any], root: Path | None = None) -> list[colab.SourceSpec]:
    return colab.resolve_sources(list(entries), sources_root=root or Path("/tmp/linescout-src"))


def test_a_preset_that_implies_a_classifier_installs_one() -> None:
    """Human-Art with no explicit override: the preset still asks for opennsfw2."""
    specs = specs_for({"preset": "human_art", "license_id": "human-art-terms"})
    assert specs[0].requires_nsfw and specs[0].uses_extractor

    plan = repro.plan_environment(specs, spec=repro.load_requirement_spec())
    assert "opennsfw2" in plan.packages, "the gate the preset implies was not asked for"
    assert "nsfw" in plan.groups and "extraction" in plan.groups
    assert "gdown" in plan.packages, "opennsfw2's own downloader was left floating"
    assert "tensorflow" not in plan.packages, "the Keras backend is a runtime, not an install"
    assert "tensorflow" in repro.RUNTIME_PACKAGES
    assert "human_art" in plan.reasons["opennsfw2"], plan.reasons["opennsfw2"]


def test_an_explicit_override_replaces_the_preset_dependency() -> None:
    """Same preset, said differently: rating comes from the source, so no gate."""
    specs = specs_for(
        {"preset": "human_art", "license_id": "human-art-terms", "sfw_method": "source_rating"}
    )
    assert not specs[0].requires_nsfw

    plan = repro.plan_environment(specs, spec=repro.load_requirement_spec(), label=False)
    assert "nsfw" not in plan.groups and "clip" not in plan.groups
    assert "opennsfw2" not in plan.packages
    assert "extraction" in plan.groups, "the extractor is still the preset's own choice"


def test_a_native_line_art_run_skips_the_extractor_group() -> None:
    specs = specs_for({"preset": "quickdraw", "license_id": "cc0-1.0"})
    plan = repro.plan_environment(
        specs, spec=repro.load_requirement_spec(), extract_line_art=False, label=False
    )
    assert "extraction" not in plan.groups and "controlnet-aux" not in plan.packages
    assert "base" in plan.groups and "display" in plan.groups


def test_dropping_a_stage_drops_its_packages() -> None:
    specs = specs_for({"preset": "amateur_drawings", "license_id": "cc-by-4.0"})
    both = repro.plan_environment(specs, spec=repro.load_requirement_spec())
    less = repro.plan_environment(
        specs, spec=repro.load_requirement_spec(), embed=False, label=False
    )
    assert "clip" in both.groups and "clip" not in less.groups
    assert "open-clip-torch" in both.packages and "open-clip-torch" not in less.packages


def test_the_plan_installs_pinned_specs_never_bare_names() -> None:
    specs = specs_for({"preset": "human_art", "license_id": "human-art-terms"})
    spec = repro.load_requirement_spec()
    assert spec is not None
    plan = repro.plan_environment(specs, spec=spec)
    assert plan.pip_specs, "nothing to install means the test runtime is a full Colab image"
    assert all("==" in item for item in plan.pip_specs), plan.pip_specs
    for item in plan.pip_specs:
        name, _, version = item.partition("==")
        assert spec.pins[name] == version
        assert name not in spec.runtime_only
    assert set(plan.pip_specs) <= {f"{name}=={spec.pins[name]}" for name in plan.packages}


def test_the_plan_reports_what_the_runtime_already_has() -> None:
    specs = specs_for({"preset": "synthetic", "license_id": "cc0"})
    plan = repro.plan_environment(specs, spec=repro.load_requirement_spec())
    assert plan.already_present, "pillow/pydantic/numpy are installed to run these tests at all"
    assert not set(plan.already_present) & set(plan.to_install)
    assert set(plan.to_install) | set(plan.already_present) == set(plan.packages)


def test_an_unresolved_source_never_silently_shrinks_the_plan() -> None:
    with pytest.raises(colab.SourceConfigurationError, match="needs a real license_id"):
        specs_for({"preset": "human_art", "license_id": "REPLACE-ME"})
    with pytest.raises(colab.SourceConfigurationError, match="unknown source preset"):
        specs_for({"preset": "not_a_dataset", "license_id": "cc0"})
    with pytest.raises(colab.SourceConfigurationError, match="empty"):
        colab.resolve_sources([], sources_root=Path("/tmp/linescout-src"))


# ---------------------------------------------------------------- recording the runtime


def test_runtime_report_describes_the_host_without_importing_torch() -> None:
    """Run in a fresh interpreter: the report must be importable on a CPU box."""
    code = (
        "import json, sys\n"
        "from linescout_ml.colab import repro\n"
        "report = repro.runtime_report()\n"
        "print(json.dumps({'torch': 'torch' in sys.modules, 'keys': sorted(report)}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["torch"] is False, "recording the runtime imported torch"
    for key in ("runtime", "python", "cuda", "packages", "torch_imported", "platform"):
        assert key in payload["keys"]


def test_the_report_never_invents_a_torch_version() -> None:
    report = repro.runtime_report()
    if report["torch_imported"] is False:
        assert isinstance(report["packages"].get("torch"), (str, type(None)))
        assert isinstance(report["cuda"], dict | str)


def test_constraints_freeze_the_local_version_suffix(tmp_path: Path) -> None:
    """``2.6.0+cu124``, not ``2.6.0`` — a CPU wheel must not satisfy the pin.

    The version is read from installed *metadata*, so this works on a machine
    where importing torch would fail, which is exactly the runtime the notebook
    is probing for.
    """
    site = tmp_path / "site"
    dist = site / "torch-2.6.0+cu124.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.6.0+cu124\n")
    constraints = tmp_path / "constraints.txt"

    sys.path.insert(0, str(site))
    try:
        lines = repro.runtime_constraints(("torch", "torchvision"), write_to=constraints)
    finally:
        sys.path.remove(str(site))

    assert lines == ["torch==2.6.0+cu124"], "only what the runtime has gets pinned"
    assert constraints.read_text().splitlines() == lines
    assert "torch" not in sys.modules, "naming a version must not import the package"

    missing = tmp_path / "nothing.txt"
    assert repro.runtime_constraints(("definitely-not-installed",), write_to=missing) == []
    assert missing.read_text() == "", "an empty constraints file is a valid empty answer"


def test_a_baseline_records_and_reports_differences(tmp_path: Path) -> None:
    report = {
        "runtime": "colab-gpu",
        "python": "3.11.11",
        "cuda": {"version": "12.4"},
        "packages": {"numpy": "1.26.4", "torch": "2.6.0+cu124"},
    }
    absent = repro.compare_runtime(report, None)
    assert set(absent) == {"baseline"}, (
        "a missing baseline must read as unrecorded, never as agreement"
    )

    same = tmp_path / "same.json"
    same.write_text(json.dumps({"packages": report["packages"]}))
    assert repro.compare_runtime(report, repro.load_baseline(same)) == {}

    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps({"packages": {"numpy": "2.0.0", "tensorflow": "2.16.1"}}))
    drift = repro.compare_runtime(report, repro.load_baseline(moved))
    assert drift["numpy"] == {"expected": "2.0.0", "actual": "1.26.4"}
    assert drift["tensorflow"]["actual"] == "absent", "a dropped preinstall is still drift"


def test_the_environment_digest_names_the_file_it_hashed(tmp_path: Path) -> None:
    missing = repro.environment_digest(tmp_path / "absent.txt")
    assert missing["sha256"] is None and missing["path"].endswith("absent.txt")

    spec = repro.environment_spec_path()
    digest = repro.environment_digest(spec)
    assert digest["sha256"] == hashlib.sha256(spec.read_bytes()).hexdigest()
    assert "requirements-colab.txt" in digest["path"]


def test_the_bundled_baseline_is_a_report_not_a_spec() -> None:
    baseline = repro.load_baseline()
    assert baseline is not None, "the notebook's comparison has nothing to compare with"
    assert baseline["packages"], "a baseline with no versions is not a record of anything"
    assert baseline.get("recorded_on"), "say what runtime this was"
    assert isinstance(baseline.get("unrecorded", []), list)


# ------------------------------------------------------------------------------ CLI


def run_cli(*argv: str) -> tuple[int, str, str]:
    """``linescout-repro`` in this interpreter, so the entry point is under test too."""
    completed = subprocess.run(
        [sys.executable, "-m", "linescout_ml.repro_cli", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def test_the_cli_reports_a_placeholder_pin_without_inventing_a_commit() -> None:
    code, out, _ = run_cli("pin", "--json")
    assert code == 0, out
    payload = json.loads(out)
    assert payload["repo"] == repro.COLAB_REPO
    assert payload["pin"] == repro.COLAB_PIN
    assert payload["url"] == f"https://github.com/{repro.COLAB_REPO}.git"
    assert payload["environment_spec"] == "/".join(repro.ENVIRONMENT_SPEC_PATH)
    # Read as a path, not as the repr of the tuple the constant really is.
    assert (REPO_ROOT / payload["environment_spec"]).is_file()
    assert payload["placeholder"] is (repro.COLAB_PIN == repro.COLAB_PIN_PLACEHOLDER)


def test_the_cli_names_the_placeholder_loudly_and_can_fail_ci_on_it() -> None:
    """Written against both pin states, so it survives the bump and still proves the gate.

    The default output depends on whether this repository's pin is final yet; the
    placeholder branch is reachable on demand with ``--pin``, which is what keeps
    the failure mode a reader of the docs is told about under test forever.
    """
    code, out, _ = run_cli("pin")
    assert code == 0, "printing the pin is not an error"
    if repro.pin_is_finalised(repro.COLAB_PIN):
        assert "immutable: True" in out, out
    else:
        assert "PLACEHOLDER" in out and "REPO_PIN" in out, out
        code, _, err = run_cli("pin", "--check")
        assert code == 1 and "PIN NOT FINALISED" in err, err

    code, out, err = run_cli("pin", "--pin", repro.COLAB_PIN_PLACEHOLDER, "--check")
    assert code == 1, out
    assert "PLACEHOLDER" in out and "PIN NOT FINALISED" in err, err


def test_a_candidate_pin_is_judged_on_the_same_two_rules(tmp_path: Path) -> None:
    """Immutable *and* not the placeholder: a branch fails even though it resolves today."""
    real_commit = "a" * 40

    code, out, _ = run_cli("pin", "--pin", real_commit, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["immutable"] is True and payload["finalised"] is True
    assert payload["placeholder"] is False

    code, out, _ = run_cli("pin", "--pin", "main", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["immutable"] is False and payload["finalised"] is False

    code, _, err = run_cli("pin", "--pin", "main", "--check")
    assert code == 1 and "PIN NOT FINALISED" in err
    assert code == 1, "a moving reference must not pass the gate CI relies on"


# ---------------------------------------------------------------- the closure hazard


def test_the_pinned_packages_reach_torch_one_level_down() -> None:
    """Why the notebook freezes the image's torch instead of merely not installing it.

    `requirements-colab.txt` never asks for torch, and that is precisely the problem:
    the resolver is asked for `open-clip-torch`, `timm`, and `controlnet-aux`, and every
    one of them requires torch, so an unconstrained install on a T4 is free to pull a
    PyPI build that has nothing to do with the driver — 81 packages and 19
    CUDA-flavoured wheels' worth, per the resolution documented in `ml/colab/README.md`.
    This reads the committed lockfile's own dependency edges, so CI proves it without
    a network or a GPU.
    """
    import tomllib

    lock = tomllib.loads((REPO_ROOT / "ml" / "uv.lock").read_text(encoding="utf-8"))
    edges = {
        package["name"]: {dep["name"] for dep in package.get("dependencies", [])}
        for package in lock["package"]
    }

    spec = repro.load_requirement_spec()
    assert spec is not None, "the committed environment spec disappeared"
    installable = set(spec.pins) - set(spec.runtime_only)
    assert installable, "every pin is runtime-only, so there is nothing to install"

    closure: set[str] = set()
    pending = list(installable)
    while pending:
        for dependency in edges.get(pending.pop(), ()):
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)

    assert "torch" in closure, (
        "no pinned package needs torch any more; the constraints file lost its reason"
    )
    assert "torch" not in installable, "the env spec started installing the thing it forbids"
    assert {"torch", "torchvision"} == set(spec.runtime_only), (
        "the runtime group is the guard; keep it exactly this"
    )
