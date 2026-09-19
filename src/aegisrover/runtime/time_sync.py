"""Ongoing clock synchronisation built on :mod:`aegisrover.runtime.clock`.

``clock.estimate`` fits a :class:`ClockModel` from one batch of samples, but a
robot clock keeps drifting afterwards, so a mapping that was good at
calibration time silently goes stale and today can only be fixed by a manual
recalibration.  :class:`ClockTracker` supervises the mapping while the system
runs:

* every sync sample is compared against the current model and the residual
  (observed minus predicted local time) is kept in a sliding window;
* the trend of those residuals estimates how fast the error is growing and
  how long remains until it crosses the tolerance — that projection is what
  requests a recalibration, before the mapping is actually unusable;
* converted timestamps come back as :class:`ConvertedStamp` with an explicit
  trust flag, and stamps stay untrusted until a recalibration passes the
  confirmation gate, so a stale mapping is never used silently.

``ClockModel``, ``ClockQuality`` and ``estimate`` are re-exported from
:mod:`aegisrover.runtime.clock` for convenience.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .clock import ClockModel, ClockQuality, estimate

__all__ = (
    'TRUSTED', 'DEGRADED', 'UNTRUSTED',
    'ClockModel', 'ClockQuality', 'estimate',
    'TrackSample', 'TrustAssessment', 'ConvertedStamp', 'ClockTracker',
)

TRUSTED = 'trusted'
DEGRADED = 'degraded'
UNTRUSTED = 'untrusted'

_LEVEL_RANK = {TRUSTED: 0, DEGRADED: 1, UNTRUSTED: 2}


@dataclass(frozen=True)
class TrackSample:
    """One sync sample and its residual against the current model.

    ``residual`` is ``None`` when the sample arrived before any model existed.
    """

    remote: float
    local: float
    residual: float | None


@dataclass(frozen=True)
class TrustAssessment:
    """Verdict on the current mapping, recomputed whenever samples arrive.

    ``observed_error`` is the worst absolute residual in the window (seconds).
    ``error_rate`` is the signed trend of the residual (seconds per second).
    ``horizon`` is the projected time from the newest sample until the trend
    crosses ±tolerance; ``math.inf`` when the trend is not approaching it.
    """

    level: str
    trustworthy: bool
    observed_error: float
    error_rate: float
    horizon: float
    needs_recalibration: bool
    reason: str
    samples: int


@dataclass(frozen=True)
class ConvertedStamp:
    """A converted timestamp annotated with its trustworthiness."""

    local: float
    trustworthy: bool
    level: str
    reason: str
    error_bound: float


class ClockTracker:
    """Supervises one robot's clock mapping between calibrations.

    ``tolerance`` is the largest acceptable conversion error in seconds; once
    the observed or projected error reaches it, converted stamps are flagged
    untrusted.  ``horizon`` is how far ahead the residual trend may look: a
    trend that crosses the tolerance within that many seconds already marks
    the mapping degraded and asks for recalibration.  A recalibration is only
    adopted when the refit passes the confirmation gate (at least
    ``confirm_samples`` used samples, RMS error within ``confirm_rms`` and
    worst-case error within ``tolerance``); until then the old model stays in
    place and stamps remain untrusted.  With ``auto_recalibrate=True`` the
    tracker refits by itself once the mapping becomes untrusted; the earlier
    trend-based warning is still reported through ``needs_recalibration`` so
    an operator can schedule a controlled recalibration before that happens.
    """

    def __init__(self, model: ClockModel | None = None, *, tolerance: float = 0.05,
                 warn_fraction: float = 0.6, window: int = 12,
                 trend_min_samples: int = 3, horizon: float = 3600.0,
                 confirm_samples: int = 4, confirm_rms: float | None = None,
                 max_residual: float = 0.05, auto_recalibrate: bool = False):
        if tolerance <= 0:
            raise ValueError('tolerance must be positive')
        if not 0 < warn_fraction < 1:
            raise ValueError('warn_fraction must be between 0 and 1')
        if window < 2 or confirm_samples < 2 or trend_min_samples < 2:
            raise ValueError('window, confirm_samples and trend_min_samples must be >= 2')
        if horizon <= 0 or max_residual <= 0:
            raise ValueError('horizon and max_residual must be positive')
        self._model = model
        self.tolerance = float(tolerance)
        self.warn_fraction = float(warn_fraction)
        self.horizon = float(horizon)
        self.confirm_samples = confirm_samples
        self.confirm_rms = float(confirm_rms) if confirm_rms is not None else self.tolerance * 0.5
        self.max_residual = float(max_residual)
        self.auto_recalibrate = auto_recalibrate
        self._trend_min_samples = trend_min_samples
        self._samples: deque[TrackSample] = deque(maxlen=window)
        self._calibration: ClockQuality | None = None
        self._last_attempt: dict | None = None
        self._assessment = self._assess()

    @classmethod
    def calibrate(cls, pairs: Iterable[tuple[float, float]], **kwargs) -> 'ClockTracker':
        """Initial calibration: fit a model from ``pairs`` and confirm it.

        Needs at least ``confirm_samples`` pairs (default 4) to confirm; with
        fewer pairs the tracker stays uncalibrated and every stamp untrusted.
        """
        tracker = cls(**kwargs)
        tracker.recalibrate(pairs)
        return tracker

    @property
    def model(self) -> ClockModel | None:
        return self._model

    @property
    def calibration(self) -> ClockQuality | None:
        """Quality of the last confirmed (re)calibration, if any."""
        return self._calibration

    @property
    def assessment(self) -> TrustAssessment:
        """Latest trust assessment, refreshed by observe()/recalibrate()."""
        return self._assessment

    def observe(self, remote: float, local: float) -> TrustAssessment:
        """Feed one sync sample and reassess the mapping."""
        residual = None if self._model is None else local - self._model.remote_to_local(remote)
        self._samples.append(TrackSample(float(remote), float(local), residual))
        self._assessment = self._assess()
        if (self.auto_recalibrate and self._assessment.level == UNTRUSTED
                and len(self._samples) >= self.confirm_samples):
            self.recalibrate()
        return self._assessment

    def recalibrate(self, pairs: Iterable[tuple[float, float]] | None = None) -> bool:
        """Refit the mapping and adopt it only if the fit confirms.

        Without ``pairs`` the sliding window is used, restricted to its newest
        ``max(confirm_samples, window // 2)`` samples: a recalibration is
        triggered because the mapping changed recently, so samples from before
        the change describe the old regime and would skew the fit.  Explicit
        ``pairs`` are used as given (they are a deliberate calibration set).

        The new model replaces the current one only when the fit passes the
        confirmation gate; until then the old model stays in place and, if the
        mapping was already judged unreliable, stamps stay untrusted.
        """
        if pairs is None:
            recent = max(self.confirm_samples, self._samples.maxlen // 2)
            block = list(self._samples)[-recent:]
            pairs = [(s.remote, s.local) for s in block]
        else:
            pairs = [(float(r), float(l)) for r, l in pairs]
        if len(pairs) < self.confirm_samples:
            self._last_attempt = {'ok': False, 'reason': 'insufficient_samples',
                                  'samples': len(pairs)}
            return False
        try:
            model, quality = estimate(pairs, robust=True, max_residual=self.max_residual)
        except ValueError as exc:
            self._last_attempt = {'ok': False, 'reason': 'fit_failed', 'detail': str(exc)}
            return False
        confirmed = (quality.used >= self.confirm_samples
                     and quality.rms_error <= self.confirm_rms
                     and quality.max_error <= self.tolerance)
        self._last_attempt = {
            'ok': confirmed,
            'reason': 'confirmed' if confirmed else 'unconfirmed',
            'used': quality.used, 'rms_error': quality.rms_error,
            'max_error': quality.max_error, 'rejected': list(quality.rejected),
        }
        if not confirmed:
            if self._model is None or self._assessment.needs_recalibration:
                reason = 'calibration_unconfirmed' if self._model is None else 'recalibration_unconfirmed'
                self._assessment = TrustAssessment(
                    UNTRUSTED, False, self._assessment.observed_error,
                    self._assessment.error_rate, 0.0, True, reason, self._assessment.samples)
            return False
        rejected = set(quality.rejected)
        self._model = model
        self._calibration = quality
        # Reseed the window with the confirmed samples (outliers dropped) so the
        # assessment immediately reflects the fresh fit instead of stale errors.
        self._samples = deque(
            (TrackSample(r, l, l - model.remote_to_local(r))
             for i, (r, l) in enumerate(pairs) if i not in rejected),
            maxlen=self._samples.maxlen)
        self._assessment = self._assess()
        return True

    def remote_to_local(self, remote_t: float) -> ConvertedStamp:
        """Convert a remote timestamp and flag whether the result is trustworthy.

        The error bound accounts for the trend: if the residual is growing,
        stamps far beyond the newest sample are flagged untrusted even when no
        fresh sample has arrived yet.  Without any model the remote stamp is
        passed through unchanged but always flagged untrusted.
        """
        assessment = self._assessment
        if self._model is None:
            return ConvertedStamp(float(remote_t), False, UNTRUSTED, assessment.reason, math.inf)
        local = self._model.remote_to_local(remote_t)
        slope, fitted = self._trend()
        last_remote = self._last_sample_remote()
        projected = abs(fitted + slope * (remote_t - last_remote)) if last_remote is not None else 0.0
        bound = max(assessment.observed_error, projected)
        if _reaches(bound, self.tolerance):
            level = UNTRUSTED
            reason = assessment.reason if assessment.level == UNTRUSTED else 'projected_tolerance_exceeded'
        elif (bound >= self.warn_fraction * self.tolerance
                and _LEVEL_RANK[assessment.level] < _LEVEL_RANK[DEGRADED]):
            level, reason = DEGRADED, 'projected_error_growing'
        else:
            level, reason = assessment.level, assessment.reason
        return ConvertedStamp(local, level != UNTRUSTED, level, reason, bound)

    def status(self) -> dict:
        a = self._assessment
        return {
            'level': a.level,
            'trustworthy': a.trustworthy,
            'reason': a.reason,
            'observed_error': a.observed_error,
            'error_rate': a.error_rate,
            'error_rate_ppm': a.error_rate * 1e6,
            'horizon': a.horizon,
            'needs_recalibration': a.needs_recalibration,
            'samples': a.samples,
            'tolerance': self.tolerance,
            'model': None if self._model is None else {
                'offset': self._model.offset,
                'drift': self._model.drift,
                'drift_ppm': self._model.drift * 1e6,
                'reference_remote': self._model.reference_remote,
            },
            'calibration': None if self._calibration is None else {
                'used': self._calibration.used,
                'rejected': list(self._calibration.rejected),
                'rms_error': self._calibration.rms_error,
                'max_error': self._calibration.max_error,
                'drift_ppm': self._calibration.drift_ppm,
            },
            'last_attempt': self._last_attempt,
        }

    # ------------------------------------------------------------------ internals
    def _assess(self) -> TrustAssessment:
        seen = [s for s in self._samples if s.residual is not None]
        if self._model is None:
            return TrustAssessment(UNTRUSTED, False, math.inf, 0.0, 0.0, True,
                                   'uncalibrated', len(self._samples))
        if not seen:
            return TrustAssessment(UNTRUSTED, False, math.inf, 0.0, 0.0, True,
                                   'no_sync_samples', len(self._samples))
        observed = max(abs(s.residual) for s in seen)
        slope, fitted = self._trend()
        horizon = _horizon(fitted, slope, self.tolerance)
        if _reaches(observed, self.tolerance):
            level, reason = UNTRUSTED, 'tolerance_exceeded'
        elif horizon <= self.horizon:
            level, reason = DEGRADED, 'trend_exceeds_within_horizon'
        elif observed >= self.warn_fraction * self.tolerance:
            level, reason = DEGRADED, 'approaching_tolerance'
        else:
            level, reason = TRUSTED, 'within_tolerance'
        needs = level == UNTRUSTED or reason == 'trend_exceeds_within_horizon'
        return TrustAssessment(level, level != UNTRUSTED, observed, slope, horizon,
                               needs, reason, len(seen))

    def _trend(self) -> tuple[float, float]:
        """Least-squares residual trend: (slope, fitted residual at newest sample)."""
        pts = [(s.remote, s.residual) for s in self._samples if s.residual is not None]
        if len(pts) < self._trend_min_samples:
            return 0.0, (pts[-1][1] if pts else 0.0)
        mt = sum(t for t, _ in pts) / len(pts)
        mr = sum(r for _, r in pts) / len(pts)
        denom = sum((t - mt) ** 2 for t, _ in pts)
        if denom == 0:
            return 0.0, pts[-1][1]
        slope = sum((t - mt) * (r - mr) for t, r in pts) / denom
        return slope, mr + slope * (pts[-1][0] - mt)

    def _last_sample_remote(self) -> float | None:
        for sample in reversed(self._samples):
            if sample.residual is not None:
                return sample.remote
        return None


def _reaches(value: float, limit: float) -> bool:
    """Boundary-safe ``value >= limit`` (floats rarely land exactly on a limit)."""
    return value >= limit or math.isclose(value, limit, rel_tol=1e-9, abs_tol=1e-12)


def _horizon(residual: float, slope: float, tolerance: float) -> float:
    """Seconds until ``residual + slope * dt`` first reaches ±tolerance."""
    if abs(residual) >= tolerance:
        return 0.0
    if slope == 0:
        return math.inf
    crossings = [(tolerance - residual) / slope, (-tolerance - residual) / slope]
    future = [dt for dt in crossings if dt > 0]
    return min(future) if future else math.inf
