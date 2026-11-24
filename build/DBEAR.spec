# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:/Users/zhang/Documents/ML_DL/Bearings_only_Localization/dbear_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/zhang/Documents/ML_DL/Bearings_only_Localization/BEAR.DEF', '.'), ('C:/Users/zhang/Documents/ML_DL/Bearings_only_Localization/build/bear.ico', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'ipykernel', 'jupyter_client', 'notebook', 'pandas', 'pytest', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'sphinx', 'docutils'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
splash = Splash(
    'C:/Users/zhang/Documents/ML_DL/Bearings_only_Localization/build/splash_DBEAR.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name='DBEAR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:/Users/zhang/Documents/ML_DL/Bearings_only_Localization/build/bear.ico'],
)
