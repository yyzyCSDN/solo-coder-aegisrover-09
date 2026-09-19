"""Recalibration supervision for robot clock mappings.

A :class:`~aegisrover.runtime.clock.ClockModel` is fitted once at calibration
time, but the remote clock keeps drifting, so the residual (observed offset
minus modelled offset) grows again until somebody recalibrates by hand.
:class:`ClockMonitor` closes that loop: it watches the residual trend of the
active model against fresh sync samples, decides when recalibration is due —
either because the tolerance is already breached or because the trend projects
a breach within a horizon — and from that moment stamps every converted
timestamp as untrusted until a new calibration passes its quality gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from aegisrover.runtime.clock import ClockModel, estimate

__all__ = ('TRUSTED', 'CALIBRATING', 'ClockHealth', 'StampedTime', 'ClockMonitor')

TRUSTED = 'trusted'
CALIBRATING = 'calibrating'


@dataclass(frozen=True)
class ClockHealth:
    state: str
    residual: float              # newest raw residual, seconds
    trend: float                 # residual slope, seconds of error per second
    recalibrate_in: float | None  # seconds until the projected breach; None if not heading for one
    uncertainty: float           # error bound for converted timestamps, seconds
    samples: int                 # residuals in the trend window
    pending: int                 # samples collected toward the pending calibration
    calibrations: int            # recalibrations confirmed so far


@dataclass(frozen=True)
class StampedTime:
    remote: float
    local: float
    trusted: bool
    uncertainty: float
    state: str


class ClockMonitor:
    """Tracks one robot's clock model and flags it when the mapping ages out.

    Sync samples (remote, local) are expected to arrive at a roughly steady
    cadence; the local clock is the stable reference, so the residual trend is
    fitted against local time. While the state is ``CALIBRATING`` every
    :meth:`stamp` result is marked untrusted, and the incoming samples are
    reused to fit a replacement model that is adopted only once its quality
    gate passes.
    """

    def __init__(self, model: ClockModel, *, tolerance: float = 0.05,
                 predict_horizon: float = 30.0, trend_window: int = 8,
                 min_trend: int = 3, spike_factor: float = 4.0,
                 min_calibration_samples: int = 6, calibration_window: int = 12,
                 confirm_rms: float | None = None):
        if tolerance <= 0:
            raise ValueError('tolerance must be positive')
        if min_trend < 2:
            raise ValueError('min_trend must be at least 2')
        self.tolerance = float(tolerance)
        self.predict_horizon = float(predict_horizon)
        self.trend_window = int(trend_window)
        self.min_trend = int(min_trend)
        self.spike_factor = float(spike_factor)
        self.min_calibration_samples = int(min_calibration_samples)
        self.calibration_window = int(calibration_window)
        self.confirm_rms = float(confirm_rms) if confirm_rms is not None else self.tolerance * 0.5
        self._model = model
        self._state = TRUSTED
        self._residuals: list[tuple[float, float]] = []
        self._pending: list[tuple[float, float]] = []
        self._calibrations = 0
        self._uncertainty = 0.0
        self._last = ClockHealth(TRUSTED, 0.0, 0.0, None, 0.0, 0, 0, 0)

    @property
    def model(self) -> ClockModel:
        return self._model

    @property
    def state(self) -> str:
        return self._state

    @property
    def health(self) -> ClockHealth:
        return self._last

    def observe(self, remote: float, local: float) -> ClockHealth:
        """Feed one sync sample and return the updated health."""
        remote = float(remote)
        local = float(local)
        residual = local - self._model.remote_to_local(remote)
        if self._state == CALIBRATING:
            self._collect(remote, local)
        else:
            self._residuals.append((local, residual))
            del self._residuals[:-self.trend_window]
            self._update_uncertainty()
            if self._due(residual):
                self._state = CALIBRATING
                self._residuals = []
                self._collect(remote, local)
        self._last = self._make_health(residual)
        return self._last

    def stamp(self, remote_t: float) -> StampedTime:
        """Convert a remote timestamp; untrusted until calibration is confirmed."""
        remote_t = float(remote_t)
        return StampedTime(remote=remote_t,
                           local=self._model.remote_to_local(remote_t),
                           trusted=self._state == TRUSTED,
                           uncertainty=self._uncertainty,
                           state=self._state)

    def begin_calibration(self) -> None:
        """Start a recalibration cycle now (e.g. on operator request)."""
        self._state = CALIBRATING
        self._residuals = []
        self._pending = []

    # -- internals -------------------------------------------------------------
    def _collect(self, remote: float, local: float) -> None:
        self._pending.append((remote, local))
        del self._pending[:-self.calibration_window]
        if len(self._pending) < self.min_calibration_samples:
            return
        try:
            model, quality = estimate(self._pending)
        except ValueError:
            return  # samples do not span enough distinct remote instants yet
        if quality.used < 3 or quality.rms_error > self.confirm_rms:
            return  # link still too noisy; keep collecting instead of confirming a bad fit
        self._model = model
        self._state = TRUSTED
        self._pending = []
        self._calibrations += 1
        self._uncertainty = quality.rms_error

    def _due(self, residual: float) -> bool:
        if abs(residual) >= self.spike_factor * self.tolerance:
            return True  # the remote clock stepped; no trend needed
        if len(self._residuals) < self.min_trend:
            return False
        slope, value = self._trend()
        if abs(value) >= self.tolerance:
            return True
        eta = self._breach_eta(slope, value)
        return eta is not None and eta <= self.predict_horizon

    def _trend(self) -> tuple[float, float]:
        """Least-squares residual slope and fitted value at the newest sample."""
        n = len(self._residuals)
        mt = sum(t for t, _ in self._residuals) / n
        mr = sum(r for _, r in self._residuals) / n
        denom = sum((t - mt) ** 2 for t, _ in self._residuals)
        slope = 0.0 if denom == 0 else sum((t - mt) * (r - mr) for t, r in self._residuals) / denom
        value = mr + slope * (self._residuals[-1][0] - mt)
        return slope, value

    def _breach_eta(self, slope: float, value: float) -> float | None:
        if abs(value) >= self.tolerance:
            return 0.0
        if slope == 0.0 or value * slope < 0.0:
            return None  # flat or moving back toward zero: no breach in sight
        return (self.tolerance - abs(value)) / abs(slope)

    def _update_uncertainty(self) -> None:
        if not self._residuals:
            return
        rms = math.sqrt(sum(r * r for _, r in self._residuals) / len(self._residuals))
        fitted = abs(self._trend()[1]) if len(self._residuals) >= self.min_trend else 0.0
        self._uncertainty = max(rms, fitted)

    def _make_health(self, residual: float) -> ClockHealth:
        if self._state == TRUSTED and len(self._residuals) >= self.min_trend:
            slope, value = self._trend()
            eta = self._breach_eta(slope, value)
        else:
            slope, eta = 0.0, None
        return ClockHealth(state=self._state, residual=residual, trend=slope,
                           recalibrate_in=eta, uncertainty=self._uncertainty,
                           samples=len(self._residuals), pending=len(self._pending),
                           calibrations=self._calibrations)
