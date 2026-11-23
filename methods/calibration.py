"""Calibrate fixed and ensemble-scaled uncertainty intervals for scalar motion predictions.

Fit the calibration constants on held-out calibration routes. Apply those constants
unchanged to a separate test set. All arrays use a common physical unit.
"""

from dataclasses import dataclass

import numpy as np


def predictive_moments(member_means, member_variances):
    """Combine within-member variance and population variance across ensemble means."""
    means = np.asarray(member_means, dtype=float)
    variances = np.asarray(member_variances, dtype=float)
    if (means.ndim != 2 or means.shape != variances.shape or 0 in means.shape
            or not np.isfinite(means).all() or not np.isfinite(variances).all() or (variances < 0).any()):
        raise ValueError("Expected finite member-by-episode means and nonnegative variances")
    return means.mean(axis=0), np.sqrt(variances.mean(axis=0) + means.var(axis=0))


def residuals_and_scale(truth, predicted_mean, predicted_sd):
    truth, mean, sd = [np.asarray(v, dtype=float) for v in (truth, predicted_mean, predicted_sd)]
    if (truth.ndim != 1 or not len(truth) or truth.shape != mean.shape or sd.shape != mean.shape
            or not all(np.isfinite(a).all() for a in (truth, mean, sd)) or (sd <= 0).any()):
        raise ValueError("Supply aligned finite scalar predictions and positive standard deviations")
    return truth - mean, sd


@dataclass(frozen=True)
class IntervalCalibration:
    fixed_radius: float
    radius_multiplier: float
    fixed_sd: float
    sd_multiplier: float

    @classmethod
    def fit(cls, truth, predicted_mean, predicted_sd):
        """Fit empirical 90% radii and Gaussian scale corrections."""
        residual, sd = residuals_and_scale(truth, predicted_mean, predicted_sd)
        magnitude = np.abs(residual)
        return cls(float(np.quantile(magnitude, .9)), float(np.quantile(magnitude / sd, .9)),
                   float(max(1e-4, np.sqrt(np.mean(residual**2)))),
                   float(max(1e-8, np.sqrt(np.mean((residual / sd)**2)))))

    def evaluate(self, truth, predicted_mean, predicted_sd):
        residual, sd = residuals_and_scale(truth, predicted_mean, predicted_sd)
        error = np.abs(residual)
        variants = {"fixed": (np.full_like(sd, self.fixed_radius), np.full_like(sd, self.fixed_sd)),
                    "ensemble_scaled": (self.radius_multiplier * sd, self.sd_multiplier * sd)}
        scores = {}
        for model, (radius, spread) in variants.items():
            difference = error - radius
            gaussian_loss = .5 * np.log(2 * np.pi) + np.log(spread) + .5 * (residual / spread)**2
            scores[model] = {
                "coverage": float(np.mean(error <= radius)),
                "mean_radius": float(radius.mean()),
                "mean_pinball_loss": float(np.mean(np.maximum(.9 * difference, -.1 * difference))),
                "mean_gaussian_nll": float(gaussian_loss.mean()),
            }
        return scores
