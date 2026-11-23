"""Start the upstream ALOHA scene with the native camera and four Warp tactile sensors."""
import builtins
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

from .assets import ConversionRecorder, apply_cached_usd_adapter
from .scene import (apply_upstream_object_fix, apply_urdf_legacy_importer_fix,
                    controlled_mesh_probe, load_flexitac_environment, warm_up_isaac6_camera)


UPSTREAM_REVISION = "dc2e1c6607a1b1a0e4b196bb4e2d4ccdc1afdcb0"
UPSTREAM_FILES = {
    "Isaacsim_tactile_env/aloha_tactile_cfg.py": "4640cbc59c443491ecc4ed0ea1f7fabff2dbf196f5d5ad6d0f2d5a865e7169f8",
    "Isaacsim_tactile_env/aloha_tactile_env.py": "773a0576c92372e517e0a64680a470bc771415d995b60e4cf43249d4a00346d3",
    "Isaacsim_tactile_env/aloha_ik_controller.py": "027f1f187ff8e13e314e5c1d2f79e42c6bec5ffa3321cbac552bd741c3a30825",
    "Isaacsim_tactile_env/viser_interface.py": "86fd82141dc6b026b373624b31959e1880ef9a33639a83482b6e222f410b918f",
    "source/isaaclab/isaaclab/sensors/warp_sdf_tactile/warp_sdf_tactile_sensor.py":
        "db153f09a790edfab9155613f731ea598c4bdce97d617f7cde788568a134ace9",
}


def _official_app():
    from isaacsim import SimulationApp
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        return SimulationApp({
            "headless": True, "hide_ui": True, "fast_shutdown": True,
            "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0, "anti_aliasing": 0,
            "width": 640, "height": 480, "limit_cpu_threads": int(os.environ.get("OMP_NUM_THREADS", 4)),
            "extra_args": ["--/renderer/multiGpu/autoEnable=false", "--/app/settings/persistent=false"],
        })
    finally:
        sys.argv = argv


def _enable_rendering(dt):
    import carb
    import omni.kit.app
    import omni.replicator.core as rep

    settings = carb.settings.get_settings()
    settings.set_bool("/isaaclab/cameras_enabled", True)
    settings.set_bool("/isaaclab/render/offscreen", True)
    manager = omni.kit.app.get_app().get_extension_manager()
    for name in ("isaacsim.core.rendering_manager", "omni.kit.viewport.utility",
                 "omni.kit.hydra_texture", "omni.kit.viewport.window"):
        if not manager.is_extension_enabled(name):
            manager.set_extension_enabled_immediate(name, True)
        if not manager.is_extension_enabled(name):
            raise RuntimeError(f"Could not enable {name}")
    from isaacsim.core.rendering_manager import RenderingManager
    RenderingManager.set_dt(dt)
    rep.orchestrator.set_capture_on_play(True)


def _check_callback():
    error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
    if error is not None:
        raise error


@contextmanager
def native_scene(repo, report, save_report, *, usd_manifest=None, prepare_assets=False):
    """Use Isaac Sim 5.1 to prepare converted assets and 6.0 for the study itself."""
    repo = Path(repo).resolve()
    lab = repo / "source/isaaclab"
    for name, expected in UPSTREAM_FILES.items():
        source = (repo / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"Upstream {name} differs from the pinned FlexiTac checkout {UPSTREAM_REVISION}")
    report["upstream_revision"] = UPSTREAM_REVISION
    sys.path.insert(0, str(lab))
    if prepare_assets:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    else:
        app = _official_app()

    env = recorder = None
    try:
        from isaacsim.core.version import get_version
        import torch
        version = ".".join(str(v) for v in get_version()[2:5])
        required = "5.1" if prepare_assets else "6.0"
        if not version.startswith(required + ".") or not torch.cuda.is_available():
            raise RuntimeError(f"This entry point requires Isaac Sim {required} and a CUDA GPU; found {version}")
        report.update(isaac_sim_version=version, gpu=torch.cuda.get_device_name(0))
        if not prepare_assets:
            from .compat import apply_isaac6_tensor_import_alias
            apply_isaac6_tensor_import_alias(version)
        env_type, config_type = load_flexitac_environment(repo)
        if prepare_assets:
            apply_urdf_legacy_importer_fix()
            recorder = ConversionRecorder(repo)
        else:
            apply_cached_usd_adapter(repo, usd_manifest)
        apply_upstream_object_fix(repo, env_type)
        cfg = config_type(headless=True, device="cuda:0", enable_camera=True)
        if not prepare_assets:
            _enable_rendering(cfg.physics_dt)
        report["launcher"] = {"kind": "Isaac Lab AppLauncher" if prepare_assets else "official Isaac Sim default SimulationApp"}
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        env = env_type(cfg, simulation_app=app)
        _check_callback()
        obs, _ = env.reset(seed=0)
        _check_callback()
        if len(env._tactile_sensors) != 4 or sorted(env._sensor_slot_order) != [0, 1, 2, 3]:
            raise RuntimeError("Expected four uniquely mapped tactile pads")
        action = np.array(obs["joint_pos"], dtype=np.float32, copy=True)
        for _ in range(60):
            obs, _, terminated, truncated, _ = env.step(action)
            _check_callback()
            pads = obs["tactile"]
            if (terminated or truncated or pads.shape != (4, 12, 32)
                    or pads.dtype != np.float32 or not np.isfinite(pads).all()
                    or pads.min() < -1e-6 or pads.max() > 1.000001):
                raise RuntimeError("Invalid observation or termination during sensor warm-up")
        report["controlled_contact"] = controlled_mesh_probe(env._tactile_sensors[0])
        if prepare_assets:
            rgb = env._camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
        else:
            rgb = warm_up_isaac6_camera(env, cfg, app, report, save_report, _check_callback)
        if rgb.shape != (cfg.camera_height, cfg.camera_width, 3) or not np.isfinite(rgb).all() or rgb.std() == 0:
            raise RuntimeError("Expected a finite, nonconstant native camera frame")
        report["camera"] = {"passed": True, "shape": list(rgb.shape)}
        save_report()
        if prepare_assets:
            recorder.write_manifest(usd_manifest, simulator_version=version, repo_commit=UPSTREAM_REVISION)
        yield env, cfg, app, _check_callback
    except Exception as error:
        report.update(status="FAILED", error=repr(error))
        save_report()
        raise
    finally:
        if recorder is not None:
            recorder.restore()
        try:
            if env is not None:
                env.close()
        finally:
            app.close()
