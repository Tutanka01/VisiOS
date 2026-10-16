import io
import threading
import unittest

from backend.agent_runner import AgentRunConfig, AgentRunner
from backend.demo import DemoSandbox
from backend.vision_agent import AgentAction, PreparedImage, VisionAgent

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def png(width=1600, height=900):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class CoordinateMappingTest(unittest.TestCase):
    def test_coord_mode_auto_detection(self):
        self.assertEqual(VisionAgent(model="qwen2.5-vl-7b-instruct").coord_mode, "pixel")
        self.assertEqual(VisionAgent(model="qwen/qwen3-vl-8b").coord_mode, "norm1000")
        self.assertEqual(VisionAgent(model="qwen3-vl", coord_mode="pixel").coord_mode, "pixel")

    def test_norm1000_maps_to_screen_pixels(self):
        agent = VisionAgent(model="qwen3-vl")
        image = PreparedImage("", "image/png", (1280, 720), (1600, 900))
        action = agent.to_screen_space(AgentAction("click", x=500, y=300), image)
        self.assertAlmostEqual(action.x, 800)
        self.assertAlmostEqual(action.y, 270)
        self.assertTrue(action.coords_absolute)
        self.assertEqual(action.model_coords, (500, 300))

    def test_pixel_mode_scales_from_model_image(self):
        # The old heuristic read (500, 300) as 0-1000 and clicked at (800, 270).
        agent = VisionAgent(model="qwen2.5-vl")
        image = PreparedImage("", "image/png", (1288, 728), (1600, 900))
        action = agent.to_screen_space(AgentAction("click", x=644, y=364), image)
        self.assertAlmostEqual(action.x, 800, delta=1)
        self.assertAlmostEqual(action.y, 450, delta=1)

    @unittest.skipIf(Image is None, "Pillow required")
    def test_pixel_mode_image_is_snapped_to_patch_grid(self):
        agent = VisionAgent(model="qwen2.5-vl", image_width=1280)
        prepared = agent.prepare_image(png(), (1600, 900))
        width, height = prepared.model_size
        self.assertEqual(width % 28, 0)
        self.assertEqual(height % 28, 0)
        self.assertEqual(prepared.screen_size, (1600, 900))

    def test_system_prompt_has_no_template_braces(self):
        prompt = VisionAgent(model="qwen3-vl").build_system_prompt((1280, 720))
        self.assertNotIn("{{", prompt)
        self.assertNotIn("__COORD_RULES__", prompt)
        self.assertIn("0-1000", prompt)


class ParsingTest(unittest.TestCase):
    def setUp(self):
        self.agent = VisionAgent(model="qwen3-vl")

    def test_flat_format_with_thought(self):
        action = self.agent._parse_action(
            '{"observation":"Firefox ouvert","thought":"Je clique","action":"double_click","x":10,"y":20}'
        )
        self.assertEqual(action._normalized_type(), "double_click")
        self.assertEqual(action.thought, "Je clique")
        self.assertEqual(action.observation, "Firefox ouvert")
        self.assertEqual((action.x, action.y), (10, 20))

    def test_coordinate_list_and_bbox(self):
        action = self.agent._parse_action('{"action":"left_click","coordinate":[100,200]}')
        self.assertEqual((action._normalized_type(), action.x, action.y), ("click", 100, 200))
        action = self.agent._parse_action('{"action":"click","bbox":[100,200,300,400]}')
        self.assertEqual((action.x, action.y), (200, 300))

    def test_answer_wrapped_in_prose_and_fence(self):
        text = 'Voici ma réponse :\n```json\n{"action":"type","text":"bonjour","submit":true,}\n```'
        action = self.agent._parse_action(text)
        self.assertEqual(action._normalized_type(), "type")
        self.assertTrue(action.submit)

    def test_last_json_object_wins_over_echoed_example(self):
        text = 'Example {"x": 1} then {"action":"key","keys":"ctrl+l"}'
        action = self.agent._parse_action(text)
        self.assertEqual(action._normalized_type(), "hotkey")
        self.assertEqual(action.text, "ctrl+l")

    def test_scroll_direction_sets_sign(self):
        action = self.agent._parse_action('{"action":"scroll","x":5,"y":5,"direction":"up","amount":4}')
        self.assertEqual(action.scroll_amount, -4)

    def test_open_url_and_run_command_and_done(self):
        self.assertEqual(self.agent._parse_action('{"action":"open_url","url":"example.com"}').url, "example.com")
        cmd = self.agent._parse_action('{"action":"run_command","command":"ls ~"}')
        self.assertEqual((cmd._normalized_type(), cmd.command), ("run_command", "ls ~"))
        done = self.agent._parse_action('{"action":"done","summary":"Fini !"}')
        self.assertEqual((done._normalized_type(), done.text), ("done", "Fini !"))

    def test_garbage_is_flagged(self):
        self.assertTrue(self.agent._parse_action("Je ne sais pas.").parse_error)


class ScriptedVision(VisionAgent):
    def __init__(self, answers):
        super().__init__(model="qwen3-vl")
        self.answers = list(answers)
        self.contexts = []

    def decide_action(self, screenshot, user_prompt, context=None, on_stream=None, should_cancel=None, **_):
        self.contexts.append(context)
        raw = self.answers.pop(0) if self.answers else '{"action":"done","summary":"ok"}'
        action = self._interpret(raw, None)
        return self.to_screen_space(action, PreparedImage("", "image/png", (1280, 720), (1600, 900)))


class FrozenSandbox(DemoSandbox):
    """A desktop on which clicks have no visible effect."""

    def send_click(self, x, y, absolute=False, button=1, repeat=1):
        self.clicks.append((x, y, absolute, button, repeat))
        return int(x), int(y)

    def __init__(self):
        super().__init__()
        self.clicks = []


@unittest.skipIf(Image is None, "Pillow required")
class RunnerLoopTest(unittest.TestCase):
    def config(self, **kwargs):
        return AgentRunConfig(trajectory_dir="/nonexistent/visios", save_screenshots=False, action_delay_seconds=0, **kwargs)

    def test_no_effect_is_reported_then_loop_guard_stops(self):
        click = '{"thought":"clic","action":"click","x":500,"y":500}'
        vision = ScriptedVision([click] * 5)
        sandbox = FrozenSandbox()
        result = AgentRunner(sandbox, vision, self.config(max_steps=8, repeat_tolerance=2)).run("test")
        self.assertEqual(result.status, "loop_guard")
        self.assertEqual(len(sandbox.clicks), 3)
        # Clicks are sent as absolute screen pixels.
        self.assertEqual(sandbox.clicks[0][:3], (800.0, 450.0, True))
        self.assertIn("did NOT change", vision.contexts[1].last_result)
        self.assertIn("MUST try a different approach", vision.contexts[2].last_result)

    def test_invalid_answers_get_feedback_then_stop(self):
        vision = ScriptedVision(["???", '{"action":"done","summary":"Fait"}'])
        result = AgentRunner(DemoSandbox(), vision, self.config()).run("test")
        self.assertEqual(result.status, "done")
        self.assertEqual(result.reason, "Fait")
        self.assertIn("did not contain a usable action", vision.contexts[1].last_result)

    def test_command_output_is_fed_back(self):
        vision = ScriptedVision(['{"action":"run_command","command":"ls"}'])
        AgentRunner(DemoSandbox(), vision, self.config()).run("test")
        self.assertIn("Desktop  Documents", vision.contexts[1].last_result)

    def test_confirmation_refused_blocks(self):
        vision = ScriptedVision(['{"action":"click","x":1,"y":1,"requires_confirmation":true}'])
        asked = []
        runner = AgentRunner(DemoSandbox(), vision, self.config(), confirm=lambda a: asked.append(a) or False)
        result = runner.run("payer la facture")
        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(asked), 1)

    def test_notes_reach_the_model_and_stop_works(self):
        vision = ScriptedVision(['{"action":"wait","seconds":0.2}'] * 10)
        events = []
        runner = AgentRunner(DemoSandbox(), vision, self.config(max_steps=10), on_event=lambda t, p: events.append(t))
        runner.add_note("plutôt sur Wikipédia")
        timer = threading.Timer(1.0, runner.stop)
        timer.start()
        result = runner.run("test")
        timer.cancel()
        self.assertEqual(result.status, "stopped")
        self.assertEqual(vision.contexts[0].user_notes, ["plutôt sur Wikipédia"])
        self.assertIn("run_finished", events)
        self.assertIn("action_planned", events)


if __name__ == "__main__":
    unittest.main()
