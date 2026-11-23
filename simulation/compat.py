"""Opt-in import-path compatibility for pinned Isaac Lab 2.x on Isaac Sim 6.

NVIDIA documents that omni.physics.tensors.impl.api moved to
omni.physics.tensors.api without changing the class identities:
https://docs.nvidia.com/isaaclab/isaac-lab/migrating-to-3-0

This module changes no tensor, quaternion, simulation or configuration behavior.
Call after SimulationApp startup, before importing the pinned asset classes.
Legacy deformable simulation is unsupported when its native tensor views are
absent. Only its eager type annotations are postponed so rigid assets can load.
"""
from __future__ import annotations

import __future__
import ast
import importlib
import importlib.abc
import importlib.machinery
from pathlib import Path
import sys
import types


_DEFORMABLE_DATA_MODULE = "isaaclab.assets.deformable_object.deformable_object_data"


class _PostponedAnnotationLoader(importlib.machinery.SourceFileLoader):
    """Compile one unchanged source with Python's standard annotation flag."""

    def get_code(self, fullname):
        if fullname != _DEFORMABLE_DATA_MODULE:
            raise ImportError(f"Unexpected module for annotation adapter: {fullname}")
        source = self.get_data(self.path)
        tree = ast.parse(source, filename=self.path)
        classes = [node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "DeformableObjectData"]
        constructors = [node for cls in classes for node in cls.body
                        if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
        annotations = [arg.annotation for node in constructors for arg in node.args.args
                       if arg.arg == "root_physx_view"]
        if len(annotations) != 1 or ast.dump(annotations[0]) != ast.dump(
                ast.parse("physx.SoftBodyView", mode="eval").body):
            raise ImportError("Pinned deformable data annotation layout changed; refusing adapter")
        # Do not read or write .pyc: normal cached bytecode lacks this flag.
        # Function bodies and numerical behavior are compiled unchanged.
        return compile(source, self.path, "exec", flags=__future__.annotations.compiler_flag,
                       dont_inherit=True)


class _PostponedAnnotationFinder(importlib.abc.MetaPathFinder):
    """Match one module and its verified installed source location only."""

    def __init__(self, source_path):
        self.source_path = source_path

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _DEFORMABLE_DATA_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != self.source_path:
            raise ImportError("Deformable annotation adapter source location mismatch")
        spec.loader = _PostponedAnnotationLoader(fullname, spec.origin)
        return spec


def _postpone_unsupported_deformable_annotations(api):
    """Allow eager package imports without claiming a missing soft-body API."""
    unavailable = [name for name in ("SoftBodyView", "SoftBodyMaterialView") if not hasattr(api, name)]
    evidence = {"applied": False, "missing_native_types": unavailable,
                "deformable_simulation_supported": not unavailable,
                "synthetic_tensor_types_created": False}
    if not unavailable:
        return evidence
    evidence["deformable_simulation_supported"] = False
    evidence["scope"] = "postpone type annotations only; no deformable simulation compatibility"
    # AppLauncher has already imported this pinned package. Do not import any
    # asset package here: its eager __init__ is precisely what needs the loader.
    lab = sys.modules.get("isaaclab")
    if lab is None or not getattr(lab, "__file__", None):
        raise RuntimeError("Pinned isaaclab package must be loaded before the annotation adapter")
    source_path = (Path(lab.__file__).resolve().parent / "assets" / "deformable_object"
                   / "deformable_object_data.py").resolve()
    if not source_path.is_file():
        raise RuntimeError("Pinned deformable data source is missing")
    if _DEFORMABLE_DATA_MODULE in sys.modules:
        annotation = sys.modules[_DEFORMABLE_DATA_MODULE].DeformableObjectData.__init__.__annotations__.get(
            "root_physx_view")
        if annotation != "physx.SoftBodyView":
            raise RuntimeError("Deformable data was imported before the required annotation adapter")
        evidence["reason"] = "type annotations already postponed"
        return evidence
    for finder in sys.meta_path:
        if isinstance(finder, _PostponedAnnotationFinder):
            if finder.source_path != source_path:
                raise RuntimeError("Deformable annotation adapter already belongs to another source")
            evidence["reason"] = "annotation loader already installed"
            evidence["applied"] = True
            return evidence
    sys.meta_path.insert(0, _PostponedAnnotationFinder(source_path))
    evidence.update({"applied": True, "module": _DEFORMABLE_DATA_MODULE, "source": str(source_path)})
    print("FLEXITAC_COMPAT: postpone unused deformable data annotations; "
          "legacy deformable simulation remains unsupported", flush=True)
    return evidence


def apply_isaac6_tensor_import_alias(simulator_version):
    """Expose the moved module at its absent legacy path, using exact identities."""
    if str(simulator_version).split(".", 1)[0] != "6":
        raise RuntimeError("The Isaac Sim 6 compatibility adapter requires actual major version 6")
    legacy = "omni.physics.tensors.impl.api"
    current = "omni.physics.tensors.api"
    try:
        original = importlib.import_module(legacy)
    except ModuleNotFoundError as error:
        # An unrelated missing dependency is not evidence of a renamed API.
        if error.name not in {legacy, "omni.physics.tensors.impl"}:
            raise
    else:
        return {"adapter": "PhysX tensor module path", "applied": False,
                "reason": "legacy module is already available", "module": original.__name__,
                "deformable_annotation_compat": _postpone_unsupported_deformable_annotations(original)}

    api = importlib.import_module(current)
    required_types = ("SimulationView", "ArticulationView", "RigidBodyView", "RigidContactView")
    missing = [name for name in required_types if not isinstance(getattr(api, name, None), type)]
    if not callable(getattr(api, "create_simulation_view", None)):
        missing.append("create_simulation_view")
    if missing:
        raise RuntimeError(f"New PhysX tensor API lacks required symbols: {missing}")

    parent = importlib.import_module("omni.physics.tensors")
    namespace = "omni.physics.tensors.impl"
    package = sys.modules.get(namespace)
    if package is None:
        if hasattr(parent, "impl"):
            raise RuntimeError("Refusing to replace an existing tensors.impl attribute")
        package = types.ModuleType(namespace)
        package.__package__ = namespace
        package.__path__ = []
        package.__spec__ = importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[namespace] = package
        parent.impl = package
    if hasattr(package, "api") and package.api is not api:
        raise RuntimeError("Refusing to replace an existing tensors.impl.api attribute")
    package.api = api
    sys.modules[legacy] = api
    assert importlib.import_module(legacy) is api
    print(f"FLEXITAC_COMPAT: {legacy} aliases {current} with unchanged class identities", flush=True)
    return {"adapter": "PhysX tensor module path", "applied": True,
            "legacy_module": legacy, "current_module": current,
            "required_symbols": [*required_types, "create_simulation_view"],
            "class_identity_preserved": True,
            "deformable_annotation_compat": _postpone_unsupported_deformable_annotations(api),
            "optional_deformable_types": {name: hasattr(api, name)
                for name in ("SoftBodyView", "SoftBodyMaterialView")}}
