"""Regression test: mathlas.__version__ tracks the packaging metadata.

The runtime __version__ literal had drifted from pyproject (1.0.1 vs 1.1.2).
__init__.py now reads the installed package metadata (single source of truth)
with a static fallback, so the two can never silently disagree again.
"""
from __future__ import annotations

import pathlib
import tomllib

import mathlas


def test_version_matches_pyproject():
    pyproject = (pathlib.Path(mathlas.__file__).resolve().parent.parent
                 / "pyproject.toml")
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert mathlas.__version__ == declared
