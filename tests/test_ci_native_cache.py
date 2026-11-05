"""Native cache hits must use current Python sources and valid native inputs."""

import csv
from collections import UserDict
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import sysconfig
from zipfile import ZipFile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/python_native_cache.py"
CI = runpy.run_path(str(SCRIPT))


@pytest.fixture
def repository(tmp_path):
    files = {
        "_rust/src/lib.rs": "// native code\n",
        "_rust/Cargo.toml": "[package]\n",
        "_rust/Cargo.lock": "# locked dependencies\n",
        "_rust/build.rs": "// build script\n",
        "_rust/kernel.h": "// included native header\n",
        ".cargo/config.toml": "# compiler configuration\n",
        "pyproject.toml": "[project]\n",
        "README.md": "Package description\n",
        ".github/actions/build-python-extension/action.yml": "# native build steps\n",
        "stratum/__init__.py": "# Python sources\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def test_python_edits_and_build_outputs_do_not_invalidate_native_cache(repository):
    before = CI["source_digest"](repository)
    (repository / "stratum/__init__.py").write_text("# changed Python source\n")
    target = repository / "_rust/target/release"
    target.mkdir(parents=True)
    (target / "libskrub_rust.so").write_bytes(b"build output")
    assert CI["source_digest"](repository) == before


@pytest.mark.parametrize("name", [
    "_rust/src/lib.rs", "_rust/Cargo.toml", "_rust/Cargo.lock", "_rust/build.rs",
    "_rust/kernel.h", ".cargo/config.toml", "pyproject.toml", "README.md",
    ".github/actions/build-python-extension/action.yml",
])
def test_native_and_metadata_changes_invalidate_cache(repository, name):
    before = CI["source_digest"](repository)
    with (repository / name).open("a") as stream:
        stream.write("\n# changed\n")
    assert CI["source_digest"](repository) != before


def test_native_file_renames_invalidate_cache(repository):
    before = CI["source_digest"](repository)
    subprocess.run(["git", "mv", "_rust/kernel.h", "_rust/renamed.h"], cwd=repository, check=True)
    assert CI["source_digest"](repository) != before


class WindowsEnvironment(UserDict):
    """Exercise Windows' uppercase iteration and case-insensitive lookups on any OS."""

    def __getitem__(self, key):
        return super().__getitem__(key.upper())

    def __setitem__(self, key, value):
        super().__setitem__(key.upper(), value)

    def __delitem__(self, key):
        super().__delitem__(key.upper())

    def __contains__(self, key):
        return super().__contains__(key.upper())


@pytest.mark.parametrize("windows_environment", [False, True])
@pytest.mark.parametrize("changed_input", ["rustc", "maturin", "ImageOS", "ImageVersion", "RUSTFLAGS"])
def test_toolchain_image_and_flags_invalidate_cache(
    repository, monkeypatch, changed_input, windows_environment,
):
    if windows_environment:
        monkeypatch.setattr(os, "environ", WindowsEnvironment(os.environ))
    check_output = subprocess.check_output
    versions = {"rustc": "rustc version one", "maturin": "maturin version one"}

    def output(command, **kwargs):
        if command[0] in versions:
            return versions[command[0]]
        return check_output(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", output)
    before = CI["cache_key"](repository)
    if changed_input in versions:
        versions[changed_input] += " updated"
    else:
        monkeypatch.setenv(changed_input, "updated")
    assert CI["cache_key"](repository) != before


def make_artifact(path, old_checkout, platform_tag=None):
    path.mkdir()
    native = path / CI["native_module"]()
    native.parent.mkdir()
    native.write_bytes(b"native payload")
    if platform_tag is None:
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    tag = f"cp311-abi3-{platform_tag}"
    wheel = path / f"stratum_ai-0.0.0.dev2-{tag}.whl"
    info = "stratum_ai-0.0.0.dev2.dist-info"
    files = {
        "stratum_ai.pth": str(old_checkout) + "\n",
        f"{info}/METADATA": "Metadata-Version: 2.1\nName: stratum-ai\nVersion: 0.0.0.dev2\n",
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n",
    }
    record = io.StringIO()
    csv.writer(record).writerows((name, "", "") for name in [*files, f"{info}/RECORD"])
    files[f"{info}/RECORD"] = record.getvalue()
    with ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return wheel


def test_regular_wheel_is_rejected_to_avoid_cached_python_sources(tmp_path):
    artifact = tmp_path / "artifact"
    wheel = make_artifact(artifact, tmp_path)
    with ZipFile(wheel, "a") as archive:
        archive.writestr("stratum/__init__.py", "# stale Python code")
    with pytest.raises(ValueError, match="editable wheel"):
        CI["editable_wheel"](artifact)


def test_missing_native_payload_is_rejected(tmp_path):
    artifact = tmp_path / "artifact"
    make_artifact(artifact, tmp_path)
    (artifact / CI["native_module"]()).unlink()
    with pytest.raises(ValueError, match="Missing native extension"):
        CI["editable_wheel"](artifact)


@pytest.mark.parametrize("platform_name, platform_tag, native_name", [
    ("linux", "linux-x86_64", "_rust_backend_native.abi3.so"),
    ("win32", "win-amd64", "_rust_backend_native.pyd"),
])
def test_platform_artifact_requires_matching_wheel_and_native_payload(
    tmp_path, monkeypatch, platform_name, platform_tag, native_name,
):
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(sysconfig, "get_platform", lambda: platform_tag)
    artifact = tmp_path / "artifact"
    wheel = make_artifact(artifact, tmp_path)
    assert CI["native_module"]() == Path("stratum") / native_name
    assert CI["editable_wheel"](artifact) == wheel

    (artifact / "stratum" / native_name).unlink()
    with pytest.raises(ValueError, match="Missing native extension"):
        CI["editable_wheel"](artifact)


@pytest.mark.parametrize("platform_name, platform_tag, wrong_tag", [
    ("linux", "linux-x86_64", "win_amd64"),
    ("win32", "win-amd64", "linux_x86_64"),
    ("win32", "win-amd64", "win_arm64"),
])
def test_wheel_from_another_platform_is_rejected(
    tmp_path, monkeypatch, platform_name, platform_tag, wrong_tag,
):
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(sysconfig, "get_platform", lambda: platform_tag)
    artifact = tmp_path / "artifact"
    make_artifact(artifact, tmp_path, platform_tag=wrong_tag)
    with pytest.raises(ValueError, match="Expected one cp311-abi3"):
        CI["editable_wheel"](artifact)


@pytest.mark.skipif(sys.platform not in ("linux", "win32"), reason="Shared artifacts target Linux and Windows")
def test_cached_artifact_installs_metadata_and_imports_current_checkout(tmp_path):
    old = tmp_path / "old-checkout"
    root = tmp_path / "current-checkout"
    for checkout, value in [(old, "old"), (root, "current")]:
        package = checkout / "stratum"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"value = {value!r}\n")
    artifact = tmp_path / "artifact"
    make_artifact(artifact, old)
    env = {
        **os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_OFFLINE": "1", "UV_LINK_MODE": "hardlink",
    }
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, env=env)
    python_path = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    python = str(venv / python_path)
    subprocess.run([
        python, "-c",
        "import runpy, sys; from pathlib import Path; "
        "runpy.run_path(sys.argv[1])['install'](Path(sys.argv[2]), Path(sys.argv[3]))",
        str(SCRIPT), str(root), str(artifact),
    ], cwd=tmp_path, check=True, env=env)
    assert (root / CI["native_module"]()).read_bytes() == b"native payload"
    # Installing the same cached wheel elsewhere must not rewrite this venv's
    # .pth through uv's shared hardlinks.
    second_venv = tmp_path / "second-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(second_venv)], check=True, env=env)
    subprocess.run([
        str(second_venv / python_path), "-c",
        "import runpy, sys; from pathlib import Path; "
        "runpy.run_path(sys.argv[1])['install'](Path(sys.argv[2]), Path(sys.argv[3]))",
        str(SCRIPT), str(old), str(artifact),
    ], cwd=tmp_path, check=True, env=env)
    # Import outside the checkout to exercise the relocated .pth, not cwd.
    probe = (
        "import stratum, json; from importlib.metadata import version; "
        "print(json.dumps([stratum.value, stratum.__file__, version('stratum-ai')]))"
    )
    result = subprocess.check_output([python, "-c", probe], cwd=tmp_path, text=True, env=env)
    assert json.loads(result) == ["current", str(root / "stratum/__init__.py"), "0.0.0.dev2"]
    # The same installed artifact must immediately see a later Python-only edit.
    (root / "stratum/__init__.py").write_text("value = 'updated Python'\n")
    result = subprocess.check_output([python, "-c", probe], cwd=tmp_path, text=True, env=env)
    assert json.loads(result)[0] == "updated Python"
