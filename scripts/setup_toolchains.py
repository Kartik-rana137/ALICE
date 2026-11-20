"""Portable C/C++ and Java toolchains for ALICE's run_code tool (Windows), inside ALICE's own tools/ folder.

  tools/gcc  GCC (MinGW-w64, UCRT) from WinLibs - the latest release on GitHub, checked against the SHA-256
             digest GitHub publishes for the file
  tools/jdk  Microsoft Build of OpenJDK 21 - checked against Microsoft's published SHA-256

No installer, no admin rights, no PATH changes: delete the folders to remove them. Both are gitignored
(/tools/). On Linux/macOS install gcc, g++ and a JDK with the system package manager instead.

    python scripts/setup_toolchains.py              # whatever is missing
    python scripts/setup_toolchains.py --only java
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
UA = {"User-Agent": "ALICE-local-study-assistant/0.1"}
GCC_RELEASES = "https://api.github.com/repos/brechtsanders/winlibs_mingw/releases/latest"
GCC_ASSET = re.compile(r"^winlibs-x86_64-posix-seh-gcc-[\d.]+-mingw-w64ucrt-[\d.]+-r\d+\.zip$")   # without LLVM
JDK_URL = "https://aka.ms/download-jdk/microsoft-jdk-21-windows-x64.zip"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def download(url: str, dest: Path, sha256: str) -> None:
    """With curl when available (follows GitHub's redirect reliably and resumes: Python's client stalled at 0 MB
    for 10 minutes on the first try), else urllib. The file is checked against `sha256` either way."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        for attempt in range(1, 6):
            r = subprocess.run([curl, "-L", "-C", "-", "--fail", "--retry", "3", "--speed-limit", "20000",
                                "--speed-time", "60", "-A", UA["User-Agent"], "-o", str(dest), url])
            if r.returncode == 0:
                break
            print(f"  curl stopped (code {r.returncode}); resuming, attempt {attempt + 1}")
        else:
            raise SystemExit(f"could not download {url}")
    else:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    if h.hexdigest().lower() != sha256.lower():
        dest.unlink()
        raise SystemExit(f"checksum mismatch for {dest.name} - deleted; try again")
    print("  checksum ok")


def unpack(zip_path: Path, target: Path) -> None:
    """Extract and move the zip's single top-level folder to `target`."""
    staging = TOOLS / "_unpack"
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  extracting to {target.relative_to(ROOT)} ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staging)
    tops = [p for p in staging.iterdir()]
    src = tops[0] if len(tops) == 1 and tops[0].is_dir() else staging
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(src), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    zip_path.unlink()


def setup_gcc() -> None:
    if (TOOLS / "gcc" / "bin" / "gcc.exe").exists():
        print("GCC: already in tools/gcc")
        return
    release = json.loads(fetch(GCC_RELEASES))
    asset = next((a for a in release["assets"] if GCC_ASSET.match(a["name"])), None)
    if asset is None or not str(asset.get("digest", "")).startswith("sha256:"):
        raise SystemExit("GCC: couldn't find the WinLibs zip (or its checksum) in the latest release")
    print(f"GCC: {asset['name']} ({asset['size'] / 1e6:.0f} MB) from WinLibs {release['tag_name']}")
    zip_path = TOOLS / "_downloads" / asset["name"]
    download(asset["browser_download_url"], zip_path, asset["digest"].split(":", 1)[1])
    unpack(zip_path, TOOLS / "gcc")


def setup_jdk() -> None:
    if (TOOLS / "jdk" / "bin" / "javac.exe").exists():
        print("Java: already in tools/jdk")
        return
    sha_line = fetch(JDK_URL + ".sha256sum.txt").decode().split()
    sha, name = sha_line[0], sha_line[-1]
    print(f"Java: {name} (Microsoft Build of OpenJDK)")
    zip_path = TOOLS / "_downloads" / name
    download(JDK_URL, zip_path, sha)
    unpack(zip_path, TOOLS / "jdk")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["c", "java"], help="just one toolchain (c covers C and C++)")
    args = ap.parse_args()
    if sys.platform != "win32":
        print("On Linux/macOS install gcc, g++ and a JDK (e.g. apt install build-essential openjdk-21-jdk); "
              "ALICE finds them on PATH.")
        return 0
    if args.only in (None, "c"):
        setup_gcc()
    if args.only in (None, "java"):
        setup_jdk()
    shutil.rmtree(TOOLS / "_downloads", ignore_errors=True)
    for exe, flag in (("gcc/bin/gcc.exe", "--version"), ("gcc/bin/g++.exe", "--version"),
                      ("jdk/bin/javac.exe", "-version")):
        p = TOOLS / exe
        if p.exists():
            out = subprocess.run([str(p), flag], capture_output=True, text=True)
            print(f"  {exe}: {(out.stdout or out.stderr).splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
