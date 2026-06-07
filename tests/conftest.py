"""Ensure tests can import the application package from the project root.

pytest does not always add the repository root to sys.path, so this file
inserts the top-level project directory before imports are resolved.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
