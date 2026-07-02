import numpy as np

from .core import NoisyBool
from .core import NoisyInt
from .core import NoisyNumber
from .core import NoisyFloat
from numpy import asarray
from numpy import isfinite
from sympy import Max
from sympy import Min


def _fold_float(values, op):
    if not values:
        raise ValueError("Requires at least one value")

    result = NoisyFloat.lift(values[0])
    for value in values[1:]:
        result = result.bin_op(value, NoisyFloat, op)
    return result


class NoisyContingencyTable:
    def __init__(self, tbl):
        # Elements should be noisy floats if they aren't noisy numbers already
        tbl = np.vectorize(lambda x: NoisyFloat.lift(x, accept=NoisyNumber))(tbl)
        # Must be 2D with positive dimensions
        assert tbl.ndim == 2 and tbl.shape[0] > 0 and tbl.shape[1] > 0
        # All observations must be finite
        assert isfinite(asarray([float(value) for value in tbl.ravel()], dtype=float)).all()
        self.tbl = tbl

    @classmethod
    def population(cls, tbl):
        return cls(tbl)

    @classmethod
    def stratified(cls, tbl):
        tbl = cls(tbl).tbl
        n_rows, n_cols = tbl.shape
        predictive = []

        for i in range(n_rows):
            row = tuple(tbl[i, :])
            row_draws = []
            remaining_total = sum(row).round_nearest()
            remaining_mass = sum(row)

            for j in range(n_cols - 1):
                cell = row[j]
                prob = cell / remaining_mass
                obs = cell.round_nearest()
                draw = NoisyInt.binomial(remaining_total, prob, obs=obs)
                remaining_total = remaining_total - draw
                remaining_mass = remaining_mass - cell
                row_draws.append(draw)

            row_draws.append(remaining_total)
            predictive.append(row_draws)

        return cls(predictive)

    def chi_squared(self):
        tbl = self.tbl
        n_rows, n_cols = tbl.shape
        row_totals = tuple(sum(tbl[i, :]) for i in range(n_rows))
        col_totals = tuple(sum(tbl[:, j]) for j in range(n_cols))
        total = sum(row_totals)

        stat = 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                expected = (row_totals[i] * col_totals[j]) / total
                stat += (tbl[i, j] - expected) ** 2 / expected

        valid = NoisyBool.lift(True)
        for row_total in row_totals:
            valid &= row_total > 0
        for col_total in col_totals:
            valid &= col_total > 0
        for cell in tbl.ravel():
            valid &= cell >= 0

        return stat.guarded(valid)

    def odds_ratio(self):
        tbl = self.tbl
        assert tbl.shape == (2, 2)

        grp0_yes, grp0_no, grp1_yes, grp1_no = tbl.ravel()
        stat = (grp0_yes * grp1_no) / (grp0_no * grp1_yes)
        valid = (grp0_yes > 0) & (grp0_no > 0) & (grp1_yes > 0) & (grp1_no > 0)
        return stat.guarded(valid)


def noisy_min(*values):
    return _fold_float(values, Min)


def noisy_max(*values):
    return _fold_float(values, Max)
