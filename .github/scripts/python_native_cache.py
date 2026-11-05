"""Cache an abi3 extension with an editable wheel, never cached Python sources.

The wheel supplies distribution metadata; its .pth is relocated on installation.
The native module lives in the current checkout, just as with maturin develop.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from zipfile import ZipFile


BUILD_INPUTS = [
    "_rust",
    "pyproject.toml",
    "README.md",
    "LICENSE*",
    ".cargo",
    "rust-toolchain*",
    ".github/workflows/python_tests.yml",
    ".github/scripts/python_native_cache.py",
    ".github/actions/build-python-extension/action.yml",
]


def native_module() -> Path:
    suffix = "pyd" if sys.platform == "win32" else "abi3.so"
    return Path(f"stratum/_rust_backend_native.{suffix}")


def source_digest(root: Path) -> str:
    """Hash tracked build inputs, including filenames to detect deletions/renames."""
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *BUILD_INPUTS], cwd=root,
    ).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(path for path in paths if path):
        content = (root / os.fsdecode(name)).read_bytes()
        digest.update(name + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def cache_key(root: Path) -> str:
    # ImageVersion also captures system linker/library updates on hosted runners.
    # Look up mixed-case names directly: Windows uppercases keys when iterating
    # os.environ, but its lookups remain case-insensitive.
    environment = {
        key: os.environ[key]
        for key in ("ImageOS", "ImageVersion", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")
        if key in os.environ
    }
    environment.update({
        key: value for key, value in os.environ.items()
        if key.startswith(("RUST", "PYO3", "CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_ENCODED_"))
    })
    inputs = {
        "source": source_digest(root),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "maturin": subprocess.check_output(["maturin", "--version"], text=True),
        "python": sys.version,
        "platform": sysconfig.get_platform(),
        "environment": environment,
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return f"python-native-v1-{sysconfig.get_platform()}-{digest}"


def editable_wheel(artifact: Path) -> Path:
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    wheels = list(artifact.glob(f"*-cp311-abi3-{platform_tag}.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one cp311-abi3 {platform_tag} wheel, found {wheels}")
    with ZipFile(wheels[0]) as archive:
        # A regular wheel would install stale Python code on a native cache hit.
        if "stratum_ai.pth" not in archive.namelist() or any(
            name.startswith("stratum/") for name in archive.namelist()
        ):
            raise ValueError("Expected an editable wheel containing metadata and a .pth only")
    if not (artifact / native_module()).is_file():
        raise ValueError(f"Missing native extension: {artifact / native_module()}")
    return wheels[0]


def build(root: Path, artifact: Path) -> None:
    import maturin

    artifact.mkdir(parents=True, exist_ok=True)
    # The PEP 660 hook creates both an in-place extension and a metadata-only
    # editable wheel without installing the project's ML dependencies.
    maturin.build_editable(str(artifact), config_settings={"build-args": "--profile release --locked"})
    destination = artifact / native_module()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / native_module(), destination)
    editable_wheel(artifact)


def install(root: Path, artifact: Path) -> None:
    wheel = editable_wheel(artifact)
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--reinstall", str(wheel)],
        check=True,
    )
    # A cached wheel's .pth points at the builder's checkout. Relocate it so
    # subprocesses also import today's sources, even outside the repository cwd.
    pth = Path(sysconfig.get_path("purelib")) / "stratum_ai.pth"
    # uv can hardlink installed files to its shared cache. Replace the file
    # instead of modifying that inode (and every other environment using it).
    relocated = pth.with_suffix(".pth.tmp")
    relocated.write_text(str(root.resolve()) + "\n", encoding="utf-8")
    relocated.replace(pth)
    shutil.copy2(artifact / native_module(), root / native_module())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["key", "build", "install"])
    parser.add_argument("--artifact", type=Path, default=Path(".cache/python-native"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    if args.command == "key":
        print(cache_key(root))
    else:
        {"build": build, "install": install}[args.command](root, args.artifact.resolve())
