from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_my_agent.py"


def _load_benchmark_module():
    module_name = "test_benchmark_my_agent_module"
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchmarkMyAgentTests(unittest.TestCase):
    def test_run_benchmarks_returns_named_results(self):
        mod = _load_benchmark_module()

        results = mod.run_benchmarks(iterations=1, warmup=0)

        names = {item["name"] for item in results}
        self.assertIn("availability_summary", names)
        self.assertIn("previous_frame_relation", names)
        self.assertIn("semantic_target_candidates", names)
        self.assertIn("recent_direction_progress_delta", names)
        self.assertIn("semantic_goal_distance", names)
        self.assertIn("semantic_target_choice", names)
        self.assertIn("semantic_direct_click_choice", names)
        self.assertIn("semantic_continuity_scale", names)
        self.assertIn("preferred_click_continuity_active", names)
        self.assertIn("semantic_click_bonus_scale", names)
        self.assertIn("template_log_bias", names)
        self.assertIn("semantic_components", names)
        self.assertIn("legal_action_mask", names)
        self.assertIn("top_legal_policy_indices", names)
        self.assertIn("candidate_scores", names)
        self.assertIn("candidate_score_map", names)
        self.assertIn("semantic_direction_bonuses", names)
        self.assertIn("semantic_candidate_action_indices", names)
        self.assertIn("click_targets_from_components", names)
        self.assertIn("rank_click_target_coords", names)
        self.assertIn("semantic_exploration_logits", names)
        self.assertIn("sample_sparse_policy_indices", names)
        self.assertIn("has_click_frontier", names)
        self.assertIn("semantic_click_targets", names)
        self.assertIn("heuristic_click_fallback_targets", names)
        self.assertIn("semantic_click_candidate_indices", names)
        self.assertIn("blocked_click_matches_coord", names)
        self.assertIn("frame_matches_previous", names)
        self.assertIn("frame_changed_since_previous", names)
        self.assertIn("recent_direction_action_index", names)
        self.assertIn("recent_direction_axis", names)
        self.assertIn("recent_click_action_index", names)
        self.assertIn("blocked_direction_action_index", names)
        self.assertIn("blocked_click_action_index", names)
        self.assertIn("click_candidate_context_map", names)
        self.assertIn("nearest_coord_within", names)
        self.assertIn("prepend_nearest_preferred_coord", names)
        self.assertIn("append_unblocked_coords", names)
        self.assertIn("semantic_click_bonus", names)
        self.assertIn("stale_wait_recovery", names)
        self.assertIn("modeled_frontier_exhausted", names)
        self.assertIn("retry_blocked_direction_after_stale_wait", names)
        self.assertIn("should_exit_warmup_early", names)
        self.assertIn("bfs_click_priority_bonus", names)
        self.assertIn("bfs_priority_bonus", names)
        self.assertIn("direction_candidate_context_map", names)
        self.assertIn("semantic_click_bonus_map", names)
        self.assertIn("heuristic_click_bonus_map", names)
        self.assertIn("wait_recovery_bonus", names)
        self.assertIn("sample_semantic_exploration_sparse", names)
        self.assertIn("engine_action_input_plain", names)
        self.assertIn("make_replay_game_and_frame", names)
        self.assertTrue(all(item["iterations"] >= 1 for item in results))


if __name__ == "__main__":
    unittest.main()
