# tests/conftest.py
import sys
from pathlib import Path

# Ensure repo root is on sys.path for `import src.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
