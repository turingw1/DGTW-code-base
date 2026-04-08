from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "debug" / ".mplconfig"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.evaluate_defect import main


if __name__ == "__main__":
    main()
