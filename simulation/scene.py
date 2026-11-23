"""Upstream environment loading, rigid object initialization and native sensor checks."""
import ast
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import types


class _LocalImportNames(ast.NodeTransformer):
    """Scope upstream generic imports without changing any checked-out files."""

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            if node.module == "utils" or node.module.startswith("utils."):
                node.module = "flexitac_local_utils" + node.module[len("utils"):]
            elif node.module == "aloha_tactile_cfg":
                node.module = "flexitac_isolated_cfg"
        return node


def load_flexitac_environment(repo: Path):
    """Load the pinned environment under private names, avoiding container utils."""
    source_dir = repo / "Isaacsim_tactile_env"
    utils_dir = source_dir / "utils"
    package_name = "flexitac_local_utils"
    if package_name in sys.modules:
        existing_paths = getattr(sys.modules[package_name], "__path__", [])
        if list(existing_paths) != [str(utils_dir)]:
            raise RuntimeError("Private FlexiTac utility namespace already belongs to another path")
    else:
        package = types.ModuleType(package_name)
        package.__path__ = [str(utils_dir)]
        package.__package__ = package_name
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
    config_path = source_dir / "aloha_tactile_cfg.py"
    spec = importlib.util.spec_from_file_location("flexitac_isolated_cfg", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = config_module  # dataclasses resolves this module name.
    spec.loader.exec_module(config_module)
    source_path = source_dir / "aloha_tactile_env.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    tree = ast.fix_missing_locations(_LocalImportNames().visit(tree))
    environment_module = types.ModuleType("flexitac_isolated_env")
    environment_module.__file__ = str(source_path)
    environment_module.__package__ = ""
    sys.modules[environment_module.__name__] = environment_module
    exec(compile(tree, str(source_path), "exec"), environment_module.__dict__)
    return environment_module.AlohaTactileEnv, config_module.AlohaTactileEnvCfg


def apply_upstream_object_fix(repo: Path, environment_class) -> None:
    """Load only the upstream patch function, without starting its Viser server."""
    source_path = repo / "Isaacsim_tactile_env" / "viser_interface.py"
    source = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    matches = [node for node in source.body if isinstance(node, ast.FunctionDef)
               and node.name == "patch_env_spawn_objects_for_socket_articulation_fix"]
    if len(matches) != 1:
        raise RuntimeError("Pinned upstream object fix was not found uniquely")
    module = ast.fix_missing_locations(_LocalImportNames().visit(ast.Module(body=matches, type_ignores=[])))
    namespace = {"os": os, "AlohaTactileEnv": environment_class, "__file__": str(source_path)}
    exec(compile(module, str(source_path), "exec"), namespace)
    namespace[matches[0].name]()


def apply_urdf_legacy_importer_fix() -> str:
    """Guard the optional setter missing from Isaac Sim 5.1's pinned URDF 2.4.31.

    The upstream converter deliberately loads 2.4.31 to preserve its old fixed
    joint merging behavior, but then unconditionally calls this newer setter.
    Keep every other statement and only skip that setter when absent.
    """
    from isaaclab.sim.converters.urdf_converter import UrdfConverter

    original = UrdfConverter._get_urdf_import_config
    if getattr(original, "_flexitac_optional_inertia_setter", False):
        return "guard optional legacy URDF set_merge_fixed_ignore_inertia"
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    matches = 0

    class GuardOptionalSetter(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal matches
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "set_merge_fixed_ignore_inertia"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "import_config"):
                matches += 1
                condition = ast.Call(func=ast.Name(id="hasattr", ctx=ast.Load()),
                                     args=[ast.Name(id="import_config", ctx=ast.Load()),
                                           ast.Constant(value="set_merge_fixed_ignore_inertia")],
                                     keywords=[])
                fallback = ast.parse(
                    'print("FLEXITAC_COMPAT: legacy URDF importer lacks optional '
                    'set_merge_fixed_ignore_inertia; keeping its native fixed-joint behavior", flush=True)'
                ).body
                return ast.copy_location(ast.If(test=condition, body=[node], orelse=fallback), node)
            return node

    tree = GuardOptionalSetter().visit(tree)
    if matches != 1:
        raise RuntimeError(f"Expected one optional URDF setter, found {matches}")
    ast.fix_missing_locations(tree)
    namespace = original.__globals__.copy()
    exec(compile(tree, inspect.getsourcefile(original) or "urdf_converter.py", "exec"), namespace)
    replacement = namespace[original.__name__]
    replacement._flexitac_optional_inertia_setter = True
    UrdfConverter._get_urdf_import_config = replacement
    return "guard optional legacy URDF set_merge_fixed_ignore_inertia"


def controlled_mesh_probe(sensor) -> dict:
    """Measure the actual live sensor at known inside/outside box distances."""
    import numpy as np
    import torch
    import trimesh
    import warp as wp
    from isaaclab.utils.warp.ops import convert_to_warp_mesh

    if not sensor.is_initialized or not all(s.is_initialized for s in sensor._pose_sensors):
        raise RuntimeError("Cannot probe an uninitialized tactile sensor")
    if not sensor.cfg.normalize_forces or not sensor.cfg.mesh_use_signed_distance:
        raise RuntimeError("Probe expects the pinned normalized, signed-distance configuration")
    original_fields = {
        name: getattr(sensor, name) for name in (
            "_target_mesh_prim_path", "_wp_mesh", "_wp_mesh_tri_indices",
            "_wp_mesh_vertex_normals", "_mesh_pos_w", "_mesh_quat_w", "_mesh_scale_w",
        )
    }
    sample_world = sensor.data.tactile_points_w[0, 0, :3].clone()
    cube = trimesh.creation.box(extents=(0.01, 0.001, 0.001))
    assert cube.is_watertight and cube.is_winding_consistent
    mesh = convert_to_warp_mesh(np.asarray(cube.vertices), np.asarray(cube.faces), sensor._wp_device)
    indices = wp.array(np.asarray(cube.faces, dtype=np.int32).flatten(), dtype=wp.int32,
                       device=sensor._wp_device)
    normals = wp.array(np.asarray(cube.vertex_normals, dtype=np.float32), dtype=wp.vec3,
                       device=sensor._wp_device)
    readings = []
    try:
        sensor._target_mesh_prim_path = "controlled_smoke_test_box"
        sensor._wp_mesh = mesh
        sensor._wp_mesh_tri_indices = indices
        sensor._wp_mesh_vertex_normals = normals
        sensor._mesh_quat_w = torch.tensor([1., 0., 0., 0.], device=sensor.device)
        sensor._mesh_scale_w = torch.ones(3, device=sensor.device)
        for depth in (-0.001, 0.0001, 0.00025):
            # At sample_world, distance to the nearest x face is exactly -depth.
            sensor._mesh_pos_w = sample_world + torch.tensor(
                [0.005 - depth, 0., 0.], device=sensor.device)
            sensor._update_buffers_impl([0])
            wp.synchronize_device(sensor._wp_device)
            value = float(sensor._data.tactile_points_w[0, 0, 3].item())
            sdf = float(sensor._sdf_out[0, 0, 0].item())
            expected = min(float(sensor.cfg.stiffness) * max(depth, 0.),
                           float(sensor.cfg.max_force)) / float(sensor.cfg.max_force)
            if not np.isclose(sdf, -depth, atol=2e-6):
                raise AssertionError(f"Controlled mesh SDF mismatch: {sdf} vs {-depth}")
            if not np.isclose(value, expected, atol=2e-3):
                raise AssertionError(f"Controlled tactile response mismatch: {value} vs {expected}")
            readings.append({"penetration_m": depth, "sdf_m": sdf,
                             "reading": value, "expected_reading": expected})
    finally:
        for name, value in original_fields.items():
            setattr(sensor, name, value)
        sensor._update_buffers_impl([0])
    return {"passed": True, "scope": "controlled mesh query on an initialized live sensor",
            "physical_grasp_validated": False, "samples": readings}


def isaac6_annotator_device_diagnostics(annotator):
    """Read metadata from the same annotator; never change its default device."""
    result = {}
    try:
        signature = inspect.signature(annotator.get_data)
        result["get_data_signature"] = str(signature)
        parameter = signature.parameters.get("device")
        supports_device = (parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        result["device_keyword_supported"] = supports_device
    except (TypeError, ValueError) as error:
        result["signature_error"] = repr(error)
        return result
    if supports_device:
        for device in ("cpu", "cuda"):
            try:
                output = annotator.get_data(device=device)
                data = output.get("data") if isinstance(output, dict) else output
                result[device] = {"is_none": data is None,
                                  "shape": list(data.shape) if hasattr(data, "shape") else None,
                                  "dtype": str(getattr(data, "dtype", None))}
            except Exception as error:
                result[device] = {"error": repr(error)}
    return result


def warm_up_isaac6_camera(env, cfg, app, report, save_report, check_callback_exception):
    """Capture a real Lab camera frame through Isaac 6's multitick renderer."""
    import numpy as np
    import threading
    import omni.graph.core as og
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.experimental.utils import stage as stage_utils

    max_attempts = 300
    timeout_seconds = 180
    started = time.monotonic()
    evidence = {"method": "RenderingManager.render + Camera.update(force_recompute=True)",
                "capture_on_play": True, "delta_time": cfg.physics_dt,
                "max_attempts": max_attempts, "timeout_seconds": timeout_seconds,
                "attempts": 0, "empty_buffer_reallocations": 0, "raw_frame_diagnostics": [],
                "sdg_graph": {"path": "/Render/PostProcess/SDGPipeline", "evaluations": 0}}
    report["camera_warmup"] = evidence
    graph_controller = og.Controller()
    sdg_graph = graph_controller.graph(evidence["sdg_graph"]["path"])
    evidence["sdg_graph"]["valid"] = bool(sdg_graph is not None and sdg_graph.is_valid())
    if not evidence["sdg_graph"]["valid"]:
        save_report()
        raise RuntimeError("Actual camera SDGPipeline graph is missing or invalid")
    annotators = env._camera._rep_registry.get("rgb", [])
    if len(annotators) != 1:
        raise RuntimeError("Camera warm-up expects exactly one actual RGB annotator")

    def fabric_time():
        try:
            prim = stage_utils.get_current_stage(backend="fabric").GetPrimAtPath("/ExternalSimulationTime")
            present = bool(prim) and prim.HasAttribute("omni:time")
            return {"present": present,
                    "time": float(prim.GetAttribute("omni:time").Get()) if present else None}
        except Exception as error:
            return {"error": repr(error)}

    if report.get("launcher", {}).get("kind") == "official Isaac Sim default SimulationApp":
        from isaacsim.sensors.camera import Camera as OfficialCamera
        prim_path = env._camera._sensor_prims[0].GetPath().pathString
        product_path = env._camera._render_product_paths[0]
        official_camera = OfficialCamera(prim_path=prim_path,
            resolution=(cfg.camera_width, cfg.camera_height),
            render_product_path=product_path, annotator_device="cpu")
        official_camera.initialize()
        env._isaac6_official_camera = official_camera
        evidence["official_camera_lifecycle"] = {"prim": prim_path, "render_product": product_path}

    evidence["fabric_time_before"] = fabric_time()
    save_report()

    def timed_out():
        report.update(status="FAILED", failed_stage="TESTING_CAMERA",
                      error=f"Isaac Sim 6 camera warm-up exceeded {timeout_seconds} seconds")
        save_report()
        print("FLEXITAC_CAMERA_TIMEOUT " + json.dumps(report), flush=True)
        # A native render call can block; the watchdog also bounds that case.
        os._exit(124)

    watchdog = threading.Timer(timeout_seconds, timed_out)
    watchdog.daemon = True
    watchdog.start()
    try:
        for attempt in range(1, max_attempts + 1):
            if not app.is_running():
                raise RuntimeError("Simulation app stopped during camera warm-up")
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError("Isaac Sim 6 camera warm-up reached its deadline")
            RenderingManager.render()
            # Isaac Sim 6's legacy Camera performs this exact evaluation in
            # its NEW_FRAME callback. Pinned Lab Camera has no such callback;
            # evaluate the real attached graph before reading its outputs.
            try:
                graph_controller.evaluate_sync(graph_id=sdg_graph)
            except Exception as error:
                evidence["sdg_graph"]["error"] = repr(error)
                save_report()
                raise
            evidence["sdg_graph"]["evaluations"] += 1
            check_callback_exception()
            evidence["attempts"] = attempt
            # Lab's first empty annotator read can allocate a permanently empty
            # buffer. Wait for real annotator pixels before asking Lab to copy
            # them; never allocate a substitute image ourselves.
            raw_output = annotators[0].get_data()
            diagnostic = {"attempt": attempt, "annotator_is_none": raw_output is None}
            raw_ready = False
            if raw_output is not None:
                raw_tensor, _ = env._camera._process_annotator_output("rgb", raw_output)
                diagnostic["annotator_shape"] = list(raw_tensor.shape)
                raw_ready = tuple(raw_tensor.shape) == (cfg.camera_height, cfg.camera_width, 3)
                if raw_ready:
                    raw_pixels = raw_tensor.detach().cpu().numpy()
                    diagnostic["annotator_finite"] = bool(np.isfinite(raw_pixels).all())
                    raw_ready = diagnostic["annotator_finite"]
            camera_data = None
            if raw_ready:
                outputs = env._camera._data.output
                empty_names = [name for name, value in outputs.items() if value.numel() == 0]
                if empty_names:
                    if len(empty_names) != len(outputs):
                        raise RuntimeError("Refusing to discard any nonempty camera output during warm-up")
                    outputs.clear()
                    evidence["empty_buffer_reallocations"] += 1
                    diagnostic["cleared_empty_buffers"] = empty_names
                # The upstream method now allocates/copies its actual annotator
                # result with its normal processing and device conventions.
                env._camera.update(dt=cfg.physics_dt, force_recompute=True)
                check_callback_exception()
                camera_data = env._camera.data.output.get("rgb")
            diagnostic["is_none"] = camera_data is None
            rgb = None
            if camera_data is not None:
                diagnostic.update(shape=list(camera_data.shape), dtype=str(camera_data.dtype),
                                  elements=int(camera_data.numel()))
                if (camera_data.ndim == 4 and camera_data.shape[0] == 1
                        and tuple(camera_data.shape[1:3]) == (cfg.camera_height, cfg.camera_width)
                        and camera_data.shape[3] in (3, 4)):
                    rgb = camera_data[0, :, :, :3].detach().cpu().numpy()
                    diagnostic["finite"] = bool(np.isfinite(rgb).all())
                    if diagnostic["finite"]:
                        diagnostic.update(min=float(rgb.min()), max=float(rgb.max()), std=float(rgb.std()))
            valid = rgb is not None and diagnostic.get("finite", False) and diagnostic.get("std", 0) > 0
            if attempt in (1, 10, max_attempts) or valid:
                diagnostic["same_annotator_devices"] = isaac6_annotator_device_diagnostics(annotators[0])
            if attempt <= 5 or attempt % 10 == 0 or valid:
                evidence["raw_frame_diagnostics"].append(diagnostic)
                evidence["fabric_time_after"] = fabric_time()
                evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
                print("FLEXITAC_CAMERA_WARMUP " + json.dumps(diagnostic), flush=True)
                save_report()
            if valid:
                evidence["passed"] = True
                # Preserve the raw frame; no environment observation fallback.
                return rgb
            # Allow asynchronous GPU completion events between render calls.
            time.sleep(0.02)
        raise RuntimeError(f"No finite nonconstant camera frame after {max_attempts} render attempts")
    finally:
        watchdog.cancel()
        evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
        evidence["fabric_time_after"] = fabric_time()
        save_report()
