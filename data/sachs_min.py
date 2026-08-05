"""Sachs (2005) cd3cd28 control condition -> (n, p) ndarray, plus a
per-coordinate skewness test.

If the download fails, get the file manually from
https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz
or export from R (SEMgraph: sachs$pkc[sachs$group == 0, ]) and pass path=.
"""

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

URL = "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz"
CACHE = Path.home() / ".cache" / "sachs.txt.gz"

NAMES = ("Raf", "Mek", "Plcg", "PIP2", "PIP3",
         "Erk", "Akt", "PKA", "PKC", "P38", "Jnk")


def load_control(path=None):
    """Return X of shape (n, p) = (853, 11), float64, raw scale."""
    if path is None:
        if not CACHE.exists():
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(URL, timeout=30) as r:
                CACHE.write_bytes(r.read())
        raw = CACHE.read_bytes()
    else:
        raw = Path(path).read_bytes()

    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

    df = df.select_dtypes("number")
    df = df.drop(columns=[c for c in df.columns
                          if str(c).lower() in {"group", "condition", "int"}],
                 errors="ignore")
    return np.ascontiguousarray(df.to_numpy(dtype=np.float64)), NAMES


def skewness_test(X, names=NAMES):
    """D'Agostino test of H0: skewness = 0, one row per coordinate."""
    z, p = stats.skewtest(X, axis=0)
    return pd.DataFrame({"skew": stats.skew(X, axis=0), "z": z, "p": p},
                        index=list(names)[:X.shape[1]])


if __name__ == "__main__":
    X = load_control()
    print("X:", X.shape, X.dtype)
    print(skewness_test(X).round(3))
