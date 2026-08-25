"""Puts sidecar/ on sys.path so tests can `import essence_models`,
`from style_analysis.palette import extract_palette`, etc. the same way the
sidecar app itself does — sidecar/ isn't an installed package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
