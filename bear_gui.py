"""BEAR (graphical version) -- bearings-only target localization.

Opens the BEAR window: enter inputs A1-A7 (defaults from BEAR.DEF), run steps
B1-B8 and inspect the results.  Packaged as ``BEAR.exe`` by ``build_exe.py``.

Usage::

    python bear_gui.py
"""

import sys

from bearloc.gui.bear_app import main

if __name__ == "__main__":
    sys.exit(main())
