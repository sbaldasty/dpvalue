"""Build a small NoisyValue-backed DataFrame and save it for the R tibble demo.

Run from the repo root with `uv run python example/make_demo_frame.py`.
Writes example/demo_frame.json, loaded by example/tidy_demo.R.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from noisyvalue.core import NoisyFloat, NoisyInt
from noisyvalue.graph import NormalNode
from noisyvalue.io import save
from noisyvalue.pandas import NoisyFloatArray, NoisyIntArray

rng = None  # fresh default RNG per draw, fine for a demo fixture

counties = ["Wayne", "Oakland", "Macomb", "Lapeer", "Livingston"]
true_counts = [420.0, 133.0, 217.0, 58.0, 71.0]

black_pop = [
    NoisyFloat.draw(v, NormalNode.create(loc=0, scale=5.0), rng=i)
    for i, v in enumerate(true_counts)
]
total_pop = [
    NoisyInt.binomial(1000, 0.4 + 0.05 * i, obs=int(400 + 30 * i))
    for i in range(len(counties))
]

df = pd.DataFrame({
    "county": counties,
    "black_pop": pd.Series(NoisyFloatArray._from_sequence(black_pop)),
    "total_pop": pd.Series(NoisyIntArray._from_sequence(total_pop)),
})

out = Path(__file__).resolve().parent / "demo_frame.json"
save(out, df)
print(f"Wrote {out}")
