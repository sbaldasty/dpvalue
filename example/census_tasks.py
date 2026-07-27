"""Male/female county populations from the 2020 DHC NMF, the noisyvalue way.

This mirrors a set of textbook differential-privacy exercises (see
tasks_census/tasks.md in uvm-plaid/census_utility): given released male and
female counts for a county, find a 95% interval for each and decide whether
the county has more men or women. The textbook version assumes the added
noise is N(0, 500) because that is a reasonable-sounding guess; it is not
the actual DHC mechanism. `get_dhc` reads the true per-cell measurement
variance from the NMF, and also applies the invariant that sharpens these
counts the most: the published PL94 table was exact in the DHC run, so
male + female is known exactly within each hispanic level. That collapses
"which is bigger" from a hypothesis test into an exact posterior
probability.

Run from the repo root with `uv run python example/census_tasks.py`. Needs
the Vermont county DHC partitions; fetch them once with:

    from noisyvalue import fetch_dhc
    fetch_dhc("county", state="VT")
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import math

from noisyvalue import get_dhc, sample_noisy_values

COUNTIES = {
    "50007": "Chittenden County, Vermont",
    "50009": "Essex County, Vermont",
}
# get_dhc filters by bare county FIPS code, not the full state+county GEOID.
COUNTY_FIPS = [geoid[2:] for geoid in COUNTIES]

# The naive textbook assumption these tasks are usually posed with, kept
# here only to show how far it is from the NMF's actual per-cell variance.
TEXTBOOK_VARIANCE = 500


def county_sex_counts():
    """Male and female `NoisyInt` counts per county, summed over hispanic."""
    df = get_dhc("county", "sex*hispanic", state="VT", county=COUNTY_FIPS)
    counts = {}
    for geoid, group in df.groupby("geoid"):
        by_sex = {sex: sum(rows["value"]) for sex, rows in group.groupby("sex")}
        counts[geoid] = by_sex
    return counts


def report(name, male, female):
    batch_male, batch_female = sample_noisy_values(male, female, n=20000, rng=0)
    lo_m, hi_m = batch_male.credible_interval()
    lo_f, hi_f = batch_female.credible_interval()
    p_more_men = (male > female).sample(20000, rng=1).draws.mean()

    textbook_moe = 1.96 * math.sqrt(TEXTBOOK_VARIANCE)

    print(f"\n{name}")
    print(f"  male:   released {int(male):>6}   posterior mean {batch_male.mean():>8.0f}"
          f"   95% CI ({lo_m:.0f}, {hi_m:.0f})"
          f"   [textbook N(0,500) would say +/-{textbook_moe:.0f}]")
    print(f"  female: released {int(female):>6}   posterior mean {batch_female.mean():>8.0f}"
          f"   95% CI ({lo_f:.0f}, {hi_f:.0f})")
    print(f"  P(more men than women) = {p_more_men:.3f}")
    print(f"  male + female always sum to the same total in every joint draw:"
          f" {len(set((batch_male.draws + batch_female.draws).round(4)))} distinct value(s)")


def task_results():
    counts = county_sex_counts()

    for geoid, name in COUNTIES.items():
        report(name, counts[geoid]["male"], counts[geoid]["female"])

    total_male = sum(counts[geoid]["male"] for geoid in COUNTIES)
    total_female = sum(counts[geoid]["female"] for geoid in COUNTIES)
    report("Chittenden + Essex combined", total_male, total_female)


if __name__ == "__main__":
    task_results()
