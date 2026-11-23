"""Observation-conditioned dynamics ensemble and receding-horizon grip selection."""
import numpy as np


def protected_grip_candidates(phases, at, count, design, *, terminal=False, transport=False):
    """Limit opening targets before a protected carrying block is committed."""
    block = phases[at:at+count]
    if terminal and np.any(block == "hold"):
        return [i for i, opening in enumerate(design["branches"].values())
                if opening <= design["reference_gripper"]]
    if transport:
        largest = max(design["branches"].values())
        lead = int(np.ceil((largest - design["reference_gripper"]) / design["maximum_opening_change_per_step"]))
        if np.any(np.isin(phases[at:at+count+lead], ("transport", "hold"))):
            return [i for i, opening in enumerate(design["branches"].values())
                    if opening <= design["reference_gripper"]]
    return None


class NativeMPC:
    def __init__(self, ensemble_file, touch, risk_weight):
        if not np.isfinite(risk_weight) or risk_weight < 0:
            raise ValueError("The risk weight must be finite and nonnegative")
        self.touch = bool(touch)
        self.risk_weight = float(risk_weight)
        arm = "early" if touch else "blind"
        with np.load(ensemble_file, allow_pickle=False) as archive:
            prefix = f"{arm}_"
            self.w = {k[len(prefix):]: archive[k] for k in archive if k.startswith(prefix)}
        self.members = len(self.w["wo"])
        self.reset()

    def reset(self):
        self._state = None
        self._touch = None
        self._frame = -1

    def _step(self, state, pads, cmd):
        # Shapes are member × candidate × feature. Each member keeps its own state and pad forecast.
        w = self.w
        commands = np.broadcast_to(cmd, (*state.shape[:2], 4))
        raw = np.concatenate([state, commands, commands[..., :3] - state[..., 6:9],
                              state[..., 6:9] - state[..., :3], pads], axis=-1)
        z = np.clip((raw - w["x_center"][:, None]) / w["x_scale"][:, None], -12, 12)
        parts = [z[..., :3], z[..., 6:20]]
        if self.touch:
            parts.append(z[..., 20:])
        x = np.concatenate(parts, axis=-1)
        first = np.tanh(x @ w["w1"] + w["b1"][:, None])
        second = np.tanh(first @ w["w2"] + w["b2"][:, None])
        out = second @ w["wo"] + w["bo"][:, None]
        next_state = state + (w["dy_center"][:, None] + w["dy_scale"][:, None] * out[..., :10])
        next_state[..., 3:6] = 0
        next_pads = np.clip(w["touch_center"][:, None] + w["touch_scale"][:, None] * out[..., 10:], 0, 1)
        return next_state, next_pads

    def _assimilate(self, history):
        plug, ee, grip, commands = (np.asarray(history[key], dtype=np.float32)
                                    for key in ("plug_xyz", "ee_xyz", "aperture", "commands"))
        n = len(plug)
        if (n < 1 or plug.shape != (n, 3) or ee.shape != (n, 3) or grip.shape != (n,)
                or commands.shape != (n - 1, 4) or self._frame >= n):
            raise ValueError("Misaligned observations or planner.reset() missing before an episode")
        if (not np.isfinite(ee).all() or not np.isfinite(grip).all() or not np.isfinite(commands).all()
                or not np.isfinite(plug[0]).all()
                or not (np.isfinite(plug).all(axis=1) | np.isnan(plug).all(axis=1)).all()):
            raise ValueError("Provide a visible anchor and wholly masked or finite subsequent plug observations")
        if self.touch:
            pads = np.asarray(history["touch"], dtype=np.float32)
            if pads.shape != (n, 8) or not np.isfinite(pads).all():
                raise ValueError("Eight observed pad features per real step are required")
        if self._frame == -1:
            self._state = np.zeros((self.members, 1, 10), dtype=np.float32)
            self._touch = np.zeros((self.members, 1, 8), dtype=np.float32)
        for frame in range(self._frame + 1, n):
            if frame:
                self._state, self._touch = self._step(self._state, self._touch, commands[frame - 1])
            if np.isfinite(plug[frame]).all():
                self._state[..., :3] = plug[frame]
            self._state[..., 6:9] = ee[frame]
            self._state[..., 9] = grip[frame]
            if self.touch:
                self._touch[:] = pads[frame]
        self._frame = n - 1
        return plug[0] - ee[0]

    def choose(self, history, plans, goal_xyz_m, *, eligible=None):
        plans = np.asarray(plans, dtype=np.float32)
        goal = np.asarray(goal_xyz_m, dtype=np.float32)
        if (plans.ndim != 3 or plans.shape[0] < 2 or plans.shape[-1] != 4
                or not 1 <= plans.shape[1] <= 60 or goal.shape != (3,)
                or not np.isfinite(plans).all() or not np.isfinite(goal).all()
                or np.any((plans[..., 3] < 0) | (plans[..., 3] > 1))):
            raise ValueError("Supply at least two valid scheduled Cartesian actions for the next 1–60 steps")
        start_offset = self._assimilate(history)
        state = np.repeat(self._state, len(plans), axis=1)
        pads = np.repeat(self._touch, len(plans), axis=1)
        peak_drift = np.zeros(state.shape[:2])
        for commands in plans.transpose(1, 0, 2):
            state, pads = self._step(state, pads, commands)
            drift = np.linalg.norm(state[..., :3] - state[..., 6:9] - start_offset, axis=-1) * 1000
            peak_drift = np.maximum(peak_drift, drift)
        goal_error = np.linalg.norm(state[..., :3] - goal, axis=-1) * 1000
        # Length-equivalent task cost. Matched rules differ solely in the multiplier of the SD.
        closure_cost = .25 * np.mean(1 - plans[..., 3], axis=1)
        costs = goal_error + .5 * np.maximum(peak_drift - 8, 0) + closure_cost
        if not np.isfinite(costs).all():
            raise ValueError("World model returned a nonfinite candidate cost")
        mean, sd = costs.mean(axis=0), costs.std(axis=0, ddof=1)
        objective = mean + self.risk_weight * sd
        action = int(np.argmin(objective)) if eligible is None else min(eligible, key=objective.__getitem__)
        record = dict(selected=action, mean_cost=mean.tolist(), sd_cost=sd.tolist(),
                      objective=objective.tolist(), horizon=plans.shape[1])
        if eligible is not None:
            record["eligible_candidates"] = list(eligible)
        return action, record
