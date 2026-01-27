
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class MiddleburyCalib:
    K0: np.ndarray
    K1: np.ndarray
    doffs: float
    baseline_m: float
    width: int
    height: int
    ndisp: int

def _parse_mat3(s: str) -> np.ndarray:
    # s like: [a b c; d e f; g h i]
    s = s.strip()
    assert s[0] == "[" and s[-1] == "]"
    body = s[1:-1]
    rows = []
    for r in body.split(";"):
        rows.append([float(x) for x in r.strip().split()])
    return np.array(rows, dtype=np.float64)

def load_middlebury_calib(path: str | Path) -> MiddleburyCalib:
    text = Path(path).read_text().strip().splitlines()
    kv = {}
    for line in text:
        if not line.strip():
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()

    K0 = _parse_mat3(kv["cam0"])
    K1 = _parse_mat3(kv["cam1"])
    doffs = float(kv["doffs"])
    baseline_mm = float(kv["baseline"])
    baseline_m = baseline_mm / 1000.0
    width = int(float(kv["width"]))
    height = int(float(kv["height"]))
    ndisp = int(float(kv["ndisp"]))

    return MiddleburyCalib(K0=K0, K1=K1, doffs=doffs, baseline_m=baseline_m, width=width, height=height, ndisp=ndisp)
