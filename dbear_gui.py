"""DBEAR (graphical version) -- target position uncertainty ellipses.

Opens the DBEAR window on a BEAR results file (default: bear_results.json in
the project folder).  Packaged as ``DBEAR.exe`` by ``build_exe.py``.

Usage::

    python dbear_gui.py [results.json]
"""

import sys

from bearloc.gui.dbear_app import main

if __name__ == "__main__":
    sys.exit(main())
