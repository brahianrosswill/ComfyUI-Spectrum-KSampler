"""Regression guard for issue #8.

`ModelPatcher.clone()` isolates `model_options` with
`comfy.utils.deepcopy_list_dict`: dict/list containers are copied, every other
object is preserved by reference. The model-patch nodes must rely on that and
nothing else.

Layering a full `copy.deepcopy(model.model_options)` on top used to duplicate
tensor and module storage, because `model_options["transformer_options"]` can
hold state objects that reference conditioning tensors and the DiT module
itself (`ModGuidanceState.dit`). With a large model that is a model-sized host
RAM spike, enough to get the ComfyUI process killed.
"""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_MODULES = ("spectrum.py", "nodes.py")


def _parse(name):
    return ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


class ModelOptionsCloneSourceTest(unittest.TestCase):
    """The nodes must not re-copy `model_options` after `model.clone()`."""

    def test_no_deepcopy_of_model_options(self):
        for name in PATCH_MODULES:
            for node in ast.walk(_parse(name)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "deepcopy"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "copy"
                ):
                    continue
                target = ast.unparse(node.args[0]) if node.args else ""
                self.assertNotIn(
                    "model_options",
                    target,
                    msg=(
                        f"{name}:{node.lineno} deep-copies {target!r}. "
                        "ModelPatcher.clone() already isolates model_options; a "
                        "full deepcopy duplicates tensor/module storage (issue #8)."
                    ),
                )

    def test_clone_model_options_helper_is_gone(self):
        for name in PATCH_MODULES:
            tree = _parse(name)
            defined = {
                n.name
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
            }
            self.assertNotIn(
                "_clone_model_options",
                defined,
                msg=f"{name} redefines the removed _clone_model_options helper (issue #8).",
            )
            called = {
                n.func.id
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            self.assertNotIn(
                "_clone_model_options",
                called,
                msg=f"{name} calls the removed _clone_model_options helper (issue #8).",
            )


class ModelOptionsCloneContractTest(unittest.TestCase):
    """Pin the isolation contract the nodes now depend on.

    Skipped when torch / comfy are not importable (bare checkout, no ComfyUI on
    the path) — the source guards above still run.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("torch not available")
        try:
            from comfy.utils import deepcopy_list_dict  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("ComfyUI not importable")

    @staticmethod
    def _model_options():
        import torch

        class _State:
            """Stand-in for ModGuidanceState: holds tensors and a module."""

            def __init__(self, dit, tensor):
                self.dit = dit
                self.tag_raw = tensor

        dit = torch.nn.Linear(8, 8)
        tensor = torch.zeros(64, dtype=torch.uint8)
        options = {
            "transformer_options": {"mod_state": _State(dit, tensor)},
            "sampler_post_cfg_function": [],
        }
        return options, dit, tensor

    def test_containers_are_independent(self):
        from comfy.utils import deepcopy_list_dict

        options, _, _ = self._model_options()
        clone = deepcopy_list_dict(options)

        clone["transformer_options"]["mod_state"] = None
        clone["sampler_post_cfg_function"].append(lambda *a: None)

        self.assertIsNotNone(options["transformer_options"]["mod_state"])
        self.assertEqual(options["sampler_post_cfg_function"], [])

    def test_tensor_and_module_storage_is_shared(self):
        from comfy.utils import deepcopy_list_dict

        options, dit, tensor = self._model_options()
        state = deepcopy_list_dict(options)["transformer_options"]["mod_state"]

        self.assertIs(state.tag_raw, tensor)
        self.assertIs(state.dit, dit)
        self.assertEqual(
            state.tag_raw.untyped_storage().data_ptr(),
            tensor.untyped_storage().data_ptr(),
        )

    def test_full_deepcopy_would_duplicate_storage(self):
        """Why the helper was removed — the behavior this guards against."""
        import copy

        options, dit, tensor = self._model_options()
        state = copy.deepcopy(options)["transformer_options"]["mod_state"]

        self.assertIsNot(state.tag_raw, tensor)
        self.assertIsNot(state.dit, dit)
        self.assertNotEqual(
            state.dit.weight.untyped_storage().data_ptr(),
            dit.weight.untyped_storage().data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
