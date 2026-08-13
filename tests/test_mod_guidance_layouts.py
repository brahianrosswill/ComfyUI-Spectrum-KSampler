import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_layouts_module():
    module_name = "spectrum_mod_guidance_layouts_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "mod_guidance_layouts.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


layouts = _load_layouts_module()


class ModGuidanceLayoutTest(unittest.TestCase):
    def test_anima_29b_manifest_lineage_is_exact(self):
        self.assertEqual(
            layouts.ANIMA_29B_SOURCE_TO_TARGET,
            (
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
                12,
                13,
                15,
                16,
                18,
                19,
                20,
                22,
                23,
                25,
                26,
                28,
                29,
                31,
                32,
                34,
                35,
                37,
                38,
                39,
            ),
        )
        self.assertEqual(
            layouts.ANIMA_29B_INSERTED_TO_SOURCE,
            (
                (2, 1),
                (5, 3),
                (8, 5),
                (11, 7),
                (14, 9),
                (17, 11),
                (21, 14),
                (24, 16),
                (27, 18),
                (30, 20),
                (33, 22),
                (36, 24),
            ),
        )
        covered_targets = set(layouts.ANIMA_29B_SOURCE_TO_TARGET)
        covered_targets.update(layouts.ANIMA_29B_INSERTION_POSITIONS)
        self.assertEqual(covered_targets, set(range(40)))

    def test_default_profile_moves_only_source_anchors(self):
        source_schedule = [0.0] * 8 + [3.0] * 19 + [0.0]

        schedule, layout_name = layouts.resolve_source_anchor_schedule(
            source_schedule,
            model_family="anima",
            model_channels=2048,
            target_block_count=40,
        )

        self.assertEqual(layout_name, "anima-2.9b-40-source-anchors")
        self.assertEqual(len(schedule), 40)
        self.assertEqual(
            [index for index, weight in enumerate(schedule) if weight == 3.0],
            [
                12,
                13,
                15,
                16,
                18,
                19,
                20,
                22,
                23,
                25,
                26,
                28,
                29,
                31,
                32,
                34,
                35,
                37,
                38,
            ],
        )
        for inserted_index in layouts.ANIMA_29B_INSERTION_POSITIONS:
            self.assertEqual(schedule[inserted_index], 0.0)

    def test_equal_depth_preserves_the_28_block_schedule(self):
        source_schedule = [float(index) for index in range(28)]

        schedule, layout_name = layouts.resolve_source_anchor_schedule(
            source_schedule,
            model_family="anima",
            model_channels=2048,
            target_block_count=28,
        )

        self.assertEqual(schedule, source_schedule)
        self.assertEqual(layout_name, "source-28-identity")

    def test_unverified_architecture_requests_native_fallback(self):
        schedule, layout_name = layouts.resolve_source_anchor_schedule(
            [3.0] * 28,
            model_family="anima",
            model_channels=3072,
            target_block_count=40,
        )

        self.assertIsNone(schedule)
        self.assertIsNone(layout_name)


class ModGuidanceProfileContractTest(unittest.TestCase):
    @staticmethod
    def _nodes_tree():
        return ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))

    def test_only_source_calibrated_profiles_request_remapping(self):
        tree = self._nodes_tree()
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MOD_W_PROFILES"
                for target in node.targets
            )
        )
        profiles = {}
        for key, value in zip(assignment.value.keys, assignment.value.values):
            kwargs = {keyword.arg: keyword.value for keyword in value.keywords}
            profiles[ast.literal_eval(key)] = kwargs["schedule_basis"].id

        self.assertEqual(
            profiles,
            {
                "step_i8_skip27": "SCHEDULE_BASIS_ANIMA_BASE_28",
                "step_i14": "SCHEDULE_BASIS_ANIMA_BASE_28",
                "uniform_w3": "SCHEDULE_BASIS_NATIVE",
            },
        )

    def test_named_profile_basis_reaches_setup_mod_guidance(self):
        tree = self._nodes_tree()
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        apply_profile_call = next(
            node
            for node in ast.walk(functions["_apply_mod_profile"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_apply_mod_guidance"
        )
        profile_keywords = {
            keyword.arg: keyword.value for keyword in apply_profile_call.keywords
        }
        forwarded_profile_basis = profile_keywords["schedule_basis"]
        self.assertIsInstance(forwarded_profile_basis, ast.Subscript)
        self.assertEqual(ast.literal_eval(forwarded_profile_basis.slice), "schedule_basis")

        setup_call = next(
            node
            for node in ast.walk(functions["_apply_mod_guidance"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setup_mod_guidance"
        )
        setup_keywords = {keyword.arg: keyword.value for keyword in setup_call.keywords}
        self.assertEqual(setup_keywords["schedule_basis"].id, "schedule_basis")


if __name__ == "__main__":
    unittest.main()
