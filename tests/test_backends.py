import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend.vision_agent import StepContext, ThinkingGuard, VisionAgent

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

ANSWER = '{"thought":"Je clique sur OK","action":"click","x":500,"y":500}'


class FakeModelServer(BaseHTTPRequestHandler):
    requests_seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeModelServer.requests_seen.append((self.path, body))
        if self.path == "/v1/chat/completions" and body["model"] == "loop-thinker":
            user_text = body["messages"][1]["content"][0]["text"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if "Stop deliberating" in user_text:
                chunks = [{"content": ANSWER}]
            else:
                chunks = [{"reasoning_content": f"Hmm, let me look again at the button number {i % 7}. "} for i in range(400)]
            try:
                for delta in chunks:
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"reasoning_content": "Je vois un bouton. "}] + [{"content": ANSWER[i : i + 10]} for i in range(0, len(ANSWER), 10)]
            for delta in chunks:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        elif self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write((json.dumps({"message": {"thinking": "Hmm. "}}) + "\n").encode())
            for i in range(0, len(ANSWER), 10):
                self.wfile.write((json.dumps({"message": {"content": ANSWER[i : i + 10]}}) + "\n").encode())
            self.wfile.write((json.dumps({"done": True}) + "\n").encode())
        else:
            self.send_response(404)
            self.end_headers()


class ThinkingGuardTest(unittest.TestCase):
    def test_budget(self):
        guard = ThinkingGuard(100)
        self.assertFalse(guard.feed("x" * 90))
        self.assertTrue(guard.feed("y" * 20))
        self.assertEqual(guard.reason, "budget")

    def test_loop_detected_but_varied_text_passes(self):
        guard = ThinkingGuard(0)
        varied = " ".join(f"word{i}" for i in range(800))
        self.assertFalse(any(guard.feed(varied[i : i + 50]) for i in range(0, len(varied), 50)))
        looping = ThinkingGuard(0)
        hits = [looping.feed("I should click the OK button to continue. ") for _ in range(100)]
        self.assertTrue(any(hits))
        self.assertEqual(looping.reason, "loop")


@unittest.skipIf(Image is None, "Pillow required")
class BackendStreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelServer)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        buffer = io.BytesIO()
        Image.new("RGB", (1600, 900)).save(buffer, format="PNG")
        cls.png = buffer.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def run_agent(self, api_mode):
        agent = VisionAgent(base_url=self.url, model="qwen/qwen3-vl-8b", api_mode=api_mode, openai_endpoint="chat_completions")
        streamed = []
        action = agent.decide_action(
            self.png, "test", screen_size=(1600, 900), context=StepContext(goal="test", step=1, max_steps=5),
            on_stream=lambda kind, delta: streamed.append((kind, delta)),
        )
        return action, streamed

    def test_openai_sse_stream(self):
        action, streamed = self.run_agent("openai")
        self.assertEqual(action._normalized_type(), "click")
        self.assertEqual((round(action.x), round(action.y)), (800, 450))
        self.assertEqual(action.thought, "Je clique sur OK")
        self.assertIn("Je vois un bouton", action.reasoning)
        self.assertIn(("thinking", "Je vois un bouton. "), streamed)
        path, body = FakeModelServer.requests_seen[-1]
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertTrue(body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_runaway_thinking_is_cut_and_answered_directly(self):
        agent = VisionAgent(base_url=self.url, model="loop-thinker", api_mode="openai", openai_endpoint="chat_completions")
        streamed = []
        action = agent.decide_action(
            self.png, "test", screen_size=(1600, 900), context=StepContext(goal="test"),
            on_stream=lambda kind, delta: streamed.append((kind, delta)),
        )
        self.assertEqual(action._normalized_type(), "click")
        self.assertTrue(agent.model_thinks)
        self.assertIn(("status", "thinking_cut:loop"), streamed)
        path, body = FakeModelServer.requests_seen[-1]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        first = [b for p, b in FakeModelServer.requests_seen if b["model"] == "loop-thinker"][0]
        self.assertEqual(first["temperature"], 0.1)  # not yet known to think
        self.assertLess(len(action.thinking), 6000)

    def test_ollama_chat_stream(self):
        action, streamed = self.run_agent("ollama")
        self.assertEqual((round(action.x), round(action.y)), (800, 450))
        self.assertGreater(len([s for s in streamed if s[0] == "text"]), 3)
        path, body = FakeModelServer.requests_seen[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(len(body["messages"][1]["images"]), 1)


if __name__ == "__main__":
    unittest.main()
