"""Build BEAR.exe and DBEAR.exe (Windows) with PyInstaller.

Usage::

    python -m pip install pyinstaller
    python build_exe.py

Output, in the ``dist`` folder::

    BEAR.exe     graphical BEAR  -- inputs A1-A7, steps B1-B8
    DBEAR.exe    graphical DBEAR -- position uncertainty ellipses
    BEAR.DEF     editable default inputs, read by BEAR.exe at start-up

Each ``.exe`` is self-contained (Python, NumPy, SciPy and Matplotlib are
inside), so the ``dist`` folder can be copied to any 64-bit Windows PC without
installing Python.  Keep the three files in the same folder: BEAR.exe reads
BEAR.DEF from its own folder, writes ``bear_results.json`` there and starts
DBEAR.exe from there.  (If BEAR.DEF is missing, a copy built into BEAR.exe is
used.)

Intermediate files go to the ``build`` folder, which can be deleted.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"

#: Executable name -> launcher script.
PROGRAMS = {"BEAR": "bear_gui.py", "DBEAR": "dbear_gui.py"}

#: Subtitle of each program's splash screen.
SPLASH_TEXT = {
    "BEAR": "Bearings-only target localization",
    "DBEAR": "Target position uncertainty ellipses",
}

#: Installed packages that are not needed and would only bloat the .exe files.
EXCLUDES = [
    "IPython", "ipykernel", "jupyter_client", "notebook", "pandas", "pytest",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "sphinx", "docutils",
]


def make_icon(path: Path) -> None:
    """Draw the program icon (platform, bearing rays, target, ellipse) as a .ico file."""
    from PIL import Image, ImageDraw

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([8, 8, size - 8, size - 8], radius=44, fill=(23, 54, 93, 255))
    platform = (60, 200)
    target = (180, 80)
    for dx, dy in ((-34, 14), (0, 0), (30, -18)):                 # bearing rays
        d.line([platform, (target[0] + dx, target[1] + dy)], fill=(200, 210, 225, 255), width=6)
    d.ellipse([118, 38, 242, 122], outline=(80, 200, 120, 255), width=10)  # uncertainty ellipse
    d.ellipse([168, 68, 192, 92], fill=(230, 70, 60, 255))                 # target
    d.ellipse([46, 186, 74, 214], fill=(90, 170, 255, 255))                # platform
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def make_splash(path: Path, name: str, subtitle: str, icon: Path) -> None:
    """Draw the "Loading..." image shown while the one-file .exe unpacks."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False):
        for face in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf"):
            try:
                return ImageFont.truetype(face, size)
            except OSError:
                continue
        return ImageFont.load_default()

    img = Image.new("RGB", (520, 200), (23, 54, 93))
    logo = Image.open(icon).convert("RGBA").resize((120, 120))
    img.paste(logo, (30, 40), logo)  # the logo's alpha channel is the paste mask
    d = ImageDraw.Draw(img)
    d.text((175, 42), name, font=font(44, bold=True), fill=(255, 255, 255))
    d.text((177, 100), subtitle, font=font(17), fill=(200, 215, 235))
    d.text((177, 140), "Loading, please wait…", font=font(15), fill=(150, 200, 160))
    img.save(path)


def build(name: str, script: str, icon: Path) -> Path:
    """Run PyInstaller for one program; return the path of the .exe."""
    import PyInstaller.__main__

    sep = os.pathsep  # PyInstaller's SOURCE<sep>DEST separator for --add-data
    args = [
        str(ROOT / script),
        "--name", name,
        "--onefile",            # a single .exe, as the assignment asks
        "--windowed",           # GUI program: no console window
        "--noconfirm",
        "--clean",
        "--distpath", str(DIST),
        "--workpath", str(BUILD / "work"),
        "--specpath", str(BUILD),
        "--icon", str(icon),
        "--splash", str(BUILD / f"splash_{name}.png"),
        "--add-data", f"{ROOT / 'BEAR.DEF'}{sep}.",
        "--add-data", f"{icon}{sep}.",
        *[f"--exclude-module={m}" for m in EXCLUDES],
    ]
    PyInstaller.__main__.run(args)
    return DIST / f"{name}.exe"


def main() -> int:
    """Build both programs and copy BEAR.DEF next to them."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed.  Install it with:\n    python -m pip install pyinstaller")
        return 1
    if sys.platform != "win32":
        print("Note: building on a non-Windows system produces executables for that system, not .exe files.")

    t0 = time.time()
    icon = BUILD / "bear.ico"
    make_icon(icon)
    for name, subtitle in SPLASH_TEXT.items():
        make_splash(BUILD / f"splash_{name}.png", name, subtitle, icon)
    exes = [build(name, script, icon) for name, script in PROGRAMS.items()]
    shutil.copy2(ROOT / "BEAR.DEF", DIST / "BEAR.DEF")

    print(f"\nBuild finished in {time.time() - t0:.0f} s.  Files in {DIST}:")
    for path in [*exes, DIST / "BEAR.DEF"]:
        print(f"  {path.name:<10} {path.stat().st_size / 1e6:8.1f} MB")
    print("\nKeep these files together in one folder and double-click BEAR.exe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
