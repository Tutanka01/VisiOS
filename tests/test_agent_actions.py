import unittest

from backend.agent_runner import AgentRunConfig, AgentRunner
from backend.vision_agent import AgentAction, VisionAgent


class VisionAgentActionParsingTest(unittest.TestCase):
    def test_parses_hotkey_action(self):
        action = VisionAgent()._parse_action(
            '{"action":"hotkey","text":"CTRL+L","confidence":0.91}'
        )

        self.assertEqual(action._normalized_type(), "hotkey")
        self.assertEqual(action.text, "CTRL+L")
        self.assertEqual(action.confidence, 0.91)

    def test_parses_drag_target(self):
        action = VisionAgent()._parse_action(
            '{"action":"drag","coordinates":{"x":100,"y":200},"to":{"x":300,"y":400},"duration":0.5}'
        )

        self.assertEqual(action._normalized_type(), "drag")
        self.assertTrue(action.has_drag_target())
        self.assertEqual(action.end_x, 300)
        self.assertEqual(action.end_y, 400)
        self.assertEqual(action.duration, 0.5)

    def test_open_browser_alias_becomes_open_app(self):
        action = VisionAgent()._parse_action('{"action":"open_browser"}')

        self.assertEqual(action._normalized_type(), "open_app")
        self.assertEqual(action.app, "browser")

    def test_done_action_is_normalized(self):
        action = VisionAgent()._parse_action('{"action":"complete","text":"La page est ouverte."}')

        self.assertEqual(action._normalized_type(), "done")


class AgentRunnerSafetyTest(unittest.TestCase):
    def test_strict_mode_blocks_sensitive_prompts(self):
        runner = AgentRunner(
            sandbox=None,
            vision=None,
            config=AgentRunConfig(safety_mode="strict"),
        )
        decision = runner._check_safety(
            "supprimer définitivement ce fichier",
            AgentAction(action_type="click", x=10, y=20),
        )

        self.assertFalse(decision.allowed)
        self.assertIn("supprimer", decision.reason)

    def test_fast_mode_allows_sensitive_prompts(self):
        runner = AgentRunner(
            sandbox=None,
            vision=None,
            config=AgentRunConfig(safety_mode="fast"),
        )
        decision = runner._check_safety(
            "supprimer définitivement ce fichier",
            AgentAction(action_type="click", x=10, y=20),
        )

        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
