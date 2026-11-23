"""Native PhysX surface material control and Cartesian command schedules."""
import hashlib
import math
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class ShapeMaterials:
    """Own the PhysX input buffers for the whole run, independently of the getter's scratch storage."""

    def __init__(self, view, sim, device, name, indices=None):
        import torch
        self.torch = torch
        self.view = view
        self.sim = sim
        self.name = name
        self.device = torch.device(device)
        self.indices = torch.tensor([0], dtype=torch.int32, device="cpu")
        self.buffers = {}
        original = self.read()
        if original.ndim != 3 or original.shape[0] != 1 or original.shape[1] == 0 or original.shape[2] != 3:
            raise RuntimeError(f"{name}: expected one instance with PhysX material triples: {original.shape}")
        self.original = original
        self.shapes = np.arange(original.shape[1]) if indices is None else np.array(indices,dtype=int)
        if (not len(self.shapes) or len(set(self.shapes))!=len(self.shapes)
                or self.shapes.min()<0 or self.shapes.max()>=original.shape[1]):
            raise RuntimeError(f"{name}: invalid shape selection {self.shapes}")
        self.other_shapes = np.setdiff1d(np.arange(original.shape[1]), self.shapes)

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def read(self):
        self.sync()
        tensor = self.view.get_material_properties()
        self.sync()
        return tensor.detach().cpu().numpy().copy()

    def apply(self, material, *, write):
        mu = (material["static_friction"], material["dynamic_friction"])
        if not (0 <= mu[1] <= mu[0] and np.isfinite(mu).all()):
            raise ValueError(f"Invalid friction coefficients: {mu}")
        if mu not in self.buffers:
            desired = np.ascontiguousarray(self.original, dtype=np.float32).copy()
            desired[0, self.shapes, 0], desired[0, self.shapes, 1] = mu
            # Allocate immutable, distinct host-side inputs for neutral, low and high.
            self.buffers[mu] = self.torch.tensor(desired, dtype=self.torch.float32, device="cpu").contiguous()
        before = float(self.sim.current_time)
        attempts = 4 if write else 1
        for attempt in range(1, attempts+1):
            if write:
                self.view.set_material_properties(self.buffers[mu], self.indices)
            actual = self.read()
            if float(self.sim.current_time) != before:
                raise RuntimeError("Physics clock advanced during material assignment/readback")
            if (actual.shape == self.original.shape and np.isfinite(actual).all()
                    and np.allclose(actual[0, self.shapes, :2], mu, rtol=0, atol=1e-5)
                    and np.array_equal(actual[0, self.other_shapes], self.original[0, self.other_shapes])
                    and np.array_equal(actual[0, :, 2], self.original[0, :, 2])):
                return {"requested": dict(material), "shape_values": actual[0, self.shapes].tolist()}
            print(f"PAIRED_MATERIAL_READBACK asset={self.name} write={write} attempt={attempt}/{attempts}"
                  f" requested={mu} actual={actual.tolist()} time={before}",flush=True)
            if attempt < attempts:
                time.sleep(0.025 * attempt)
        raise RuntimeError(f"{self.name} PhysX material did not match {mu}: {actual.tolist()}")


def right_contact_shapes(robot, required):
    """Derive articulation table offsets the same way as IsaacLab's material randomizer."""
    view = robot.root_physx_view
    links = []
    offset = 0
    for path in view.link_paths[0]:
        body_view = robot._physics_sim_view.create_rigid_body_view(path)
        count = int(body_view.max_shapes)
        if path.rsplit("/",1)[-1] in required:
            if count <= 0:
                raise RuntimeError(f"Selected right finger/pad has no collision shapes: {path}")
            links.append(dict(link_path=path, shape_indices=list(range(offset,offset+count))))
        offset += count
    if (offset != view.max_shapes or len(links)!=len(required)
            or {r["link_path"].rsplit("/",1)[-1] for r in links} != set(required)):
        raise RuntimeError(f"Failed to locate exact right contact shapes: {links}, {offset}/{view.max_shapes}")
    return links


def command_schedule(config, family, grasp, left_position, left_quat, right_quat):
    """Build UI commands from world-frame target positions, including pre-lift excitation."""
    dt = config["physics_dt_s"]
    seconds = config["seconds"]
    grip = family["gripper"]
    ready = grasp + config["approach_delta_m"]
    preload = grasp + np.array([0, 0, config["preprobe_rise_m"]])
    shear = preload + np.r_[family["probe_xy_m"], 0]
    lift = shear + [0, 0, family["lift_m"] - config["preprobe_rise_m"]]
    motions = [
        ("approach", grasp, 1., None), ("settle", grasp, 1., seconds["settle"]),
        ("grip", grasp, grip, seconds["grip"]), ("preprobe", preload, grip, seconds["preprobe"]),
        ("probe", shear, grip, seconds["probe"]), ("probe_hold", shear, grip, seconds["probe_hold"]),
        ("lift", lift, grip, None),
    ]
    for xy in family["path_xy_m"]:
        motions.append(("transport", lift + np.r_[xy, 0], grip, None))
    end = motions[-1][1]
    motions.extend([("hold", end, grip, seconds["hold"]), ("release", end, 1., seconds["release"]),
                    ("retract", end + config["retract_delta_m"], 1., None)])
    previous, previous_gripper = ready, 1.
    rows, segments, phases, ids = [], [], [], []
    for sid, (phase, target, aperture, duration) in enumerate(motions):
        if duration is None:
            duration = np.linalg.norm(target - previous) / family["speed_m_s"]
        steps = max(1, math.ceil(duration / dt))
        segments.append(dict(segment_id=sid, phase=phase, start_step=len(rows), steps=steps,
                             target_position_w_m=target.tolist(), target_gripper=float(aperture)))
        for j in range(1, steps+1):
            f = j / steps
            pos = previous + f * (target-previous)
            g = previous_gripper + f * (aperture-previous_gripper)
            rows.append(np.r_[left_position, left_quat, 1., pos, right_quat, g])
            phases.append(phase)
            ids.append(sid)
        previous, previous_gripper = target, aperture
    return np.asarray(rows, dtype=np.float32), np.asarray(phases), np.asarray(ids), segments
