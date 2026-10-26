"""Lets the tests run from any folder: the analyser handler, the vendored algorithm and the v1 adapter are importable,
and the working folder is algorithm/ because the handler reads configs/ relative to it."""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "analyser_lambda"), str(ROOT / "algorithm"), str(ROOT / "src"), str(ROOT.parent)]


@pytest.fixture(autouse=True)
def _algorithm_cwd(monkeypatch):
    monkeypatch.chdir(ROOT / "algorithm")
