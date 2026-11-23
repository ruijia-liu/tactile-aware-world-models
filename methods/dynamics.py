"""Paired MLPs for residual robot/object dynamics and auxiliary pad forecasting."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

TOUCH_WEIGHT = 0.25

@contextmanager
def atomic_output(path):
    """Publish a complete file even if a training run is interrupted."""
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.close()
            temporary.replace(path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def features(state, action, tactile):
    """Relative distances expose command tracking and gripper/object geometry."""
    command_error = action[:, :3] - state[:, 6:9]
    relative_ee = state[:, 6:9] - state[:, :3]
    return np.concatenate([state, action, command_error, relative_ee, tactile], axis=1)


def center_scale(rows):
    center = rows.mean(axis=0, dtype=np.float64)
    scale = rows.std(axis=0, dtype=np.float64).clip(min=1e-7)
    return center.astype(np.float32), scale.astype(np.float32)


def tactile_features(taxels):
    """Two right pads; integrated normalized response and taxel-map centroids.

    Tail order is (pad, row, column). Center x is column/31, y is row/11.
    Zero response has zero center coordinates. Every summary lies in [0,1].
    """
    if taxels.ndim != 4 or taxels.shape[1:] != (4, 12, 32):
        raise ValueError("Expected four 12-by-32 normalized pad maps per observation")
    right = taxels[:, 2:4]
    if not np.isfinite(right).all() or np.min(right) < 0 or np.max(right) > 1:
        raise ValueError("Expected normalized sensor readings in [0,1]")
    mass = right.sum(axis=(2, 3), dtype=np.float64)
    col_moment = right.sum(axis=2, dtype=np.float64) @ np.linspace(0, 1, 32)
    row_moment = right.sum(axis=3, dtype=np.float64) @ np.linspace(0, 1, 12)
    center_x = np.divide(col_moment, mass, where=mass > 0, out=np.zeros_like(mass))
    center_y = np.divide(row_moment, mass, where=mass > 0, out=np.zeros_like(mass))
    summary = np.stack([right.max(axis=(2, 3)), mass/384, center_x, center_y], axis=-1)
    return np.ascontiguousarray(summary.reshape(len(taxels), -1), dtype=np.float32)

class DynamicsModel:
    def __init__(self, name, params):
        self.name = name
        self.p = params
        self.state_dim = len(params["dy_center"])

    def inputs(self, state, pads, action):
        raw = features(state, action, pads)
        z = np.clip((raw - self.p["x_center"]) / self.p["x_scale"], -12, 12)
        columns = [z[:, :3], z[:, 6:20]]
        if self.name == "early":
            columns.append(z[:, 20:])
        return np.concatenate(columns, axis=1)

    def forward(self, x):
        p = self.p
        h = np.tanh(x @ p["w1"] + p["b1"])
        joint = np.tanh(h @ p["w2"] + p["b2"])
        return h, joint, joint @ p["wo"] + p["bo"]

    def step(self, state, pads, action):
        p = self.p
        _, _, out = self.forward(self.inputs(state, pads, action))
        state = state + (p["dy_center"] + p["dy_scale"] * out[:, :self.state_dim])
        state[:, 3:6] = 0
        next_pads = np.clip(p["touch_center"] + p["touch_scale"] * out[:, self.state_dim:], 0, 1)
        return state, next_pads, None


def weights(seed, layer, fan_in, width, scale=1.0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, layer]))
    return rng.normal(0, scale/np.sqrt(fan_in), (fan_in, width)).astype(np.float32)


def initialize(arm, seed, scalers):
    input_width, hidden_width = {"blind": (17, 70), "early": (25, 64)}[arm]
    params = dict(scalers, w1=weights(seed, 1, input_width, hidden_width),
                  b1=np.zeros(hidden_width, np.float32), w2=weights(seed, 3, hidden_width, 64),
                  b2=np.zeros(64, np.float32), wo=weights(seed, 4, 64, 18, .04),
                  bo=np.zeros(18, np.float32))
    return DynamicsModel(arm, params)


def gradients(model, x, y):
    first, joint, output = model.forward(x)
    error = output - y
    state_error, pad_error = error[:, :model.state_dim], error[:, model.state_dim:]
    loss = np.mean(state_error**2) + TOUCH_WEIGHT * np.mean(pad_error**2)
    grad = np.empty_like(output)
    grad[:, :model.state_dim] = 2 * state_error / state_error.size
    grad[:, model.state_dim:] = (2 * TOUCH_WEIGHT / pad_error.size) * pad_error
    grad_joint = (grad @ model.p["wo"].T) * (1 - joint**2)
    grad_first = (grad_joint @ model.p["w2"].T) * (1 - first**2)
    return float(loss), dict(wo=joint.T @ grad, bo=grad.sum(axis=0), w2=first.T @ grad_joint,
                             b2=grad_joint.sum(axis=0), w1=x.T @ grad_first, b1=grad_first.sum(axis=0))


def scalers_for(rows):
    state, action, touch, delta, next_touch = rows
    xc, xs = center_scale(features(state, action, touch))
    dc, ds = center_scale(delta)
    tc, ts = center_scale(next_touch)
    return dict(x_center=xc,x_scale=xs,dy_center=dc,dy_scale=ds,touch_center=tc,touch_scale=ts)


def supervised_arrays(model, rows):
    state, action, touch, delta, next_touch = rows
    p = model.p
    targets = np.column_stack([(delta - p["dy_center"]) / p["dy_scale"],
                               (next_touch - p["touch_center"]) / p["touch_scale"]])
    return model.inputs(state, touch, action), targets
