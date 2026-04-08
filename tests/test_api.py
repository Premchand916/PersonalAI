import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.server import app


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def invoke(self, state):
        if self.error is not None:
            raise self.error
        return self.result


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint_returns_ok(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "service": "PersonalAI"},
        )

    def test_root_redirects_to_docs(self):
        response = self.client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/docs")

    def test_blank_task_is_rejected(self):
        response = self.client.post(
            "/research",
            json={"task": "   ", "post_to_telegram": False},
        )

        self.assertEqual(response.status_code, 422)

    def test_research_returns_structured_response(self):
        result = {
            "final_status": "completed",
            "research_summary": "Summary text",
            "twitter_thread": ["1/ One", "2/ Two"],
            "search_results": [{"url": "https://example.com"}],
            "scraped_content": ["content"],
            "error": None,
        }

        with patch("api.server.get_graph", return_value=FakeGraph(result=result)):
            response = self.client.post(
                "/research",
                json={"task": "demo task", "post_to_telegram": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["thread"], ["1/ One", "2/ Two"])
        self.assertEqual(response.json()["sources_found"], 1)
        self.assertEqual(response.json()["sources_used"], 1)

    def test_research_returns_http_500_when_graph_raises(self):
        with patch(
            "api.server.get_graph",
            return_value=FakeGraph(error=RuntimeError("boom")),
        ):
            response = self.client.post(
                "/research",
                json={"task": "demo task", "post_to_telegram": False},
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Pipeline failed: boom", response.json()["detail"])
