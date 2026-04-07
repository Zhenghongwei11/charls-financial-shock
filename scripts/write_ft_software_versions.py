#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "results" / "qc" / "ft_software_versions.tsv"


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"component": "python", "version": sys.version.split()[0], "detail": sys.version.replace("\n", " ")},
        {"component": "platform", "version": platform.platform(), "detail": platform.platform()},
        {"component": "pandas", "version": pkg_version("pandas"), "detail": ""},
        {"component": "numpy", "version": pkg_version("numpy"), "detail": ""},
        {"component": "statsmodels", "version": pkg_version("statsmodels"), "detail": ""},
        {"component": "matplotlib", "version": pkg_version("matplotlib"), "detail": ""},
        {"component": "requests", "version": pkg_version("requests"), "detail": ""},
    ]
    pd.DataFrame(rows).to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

