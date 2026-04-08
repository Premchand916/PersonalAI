import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.publisher_agent import MAX_POST_LENGTH, publisher_agent_node
from agents.research_agent import research_agent_node
from agents.search_agent import _parse_queries, search_agent_node
from tools.ollama_client import LLMInvocationError, invoke_prompt
from tools.scraper import ScrapeSuccess


class AgentTests(unittest.TestCase):
    def test_parse_queries_falls_back_for_invalid_output(self):
        queries = _parse_queries("not a list", "fallback query")

        self.assertEqual(queries, ["fallback query"])

    def test_search_agent_uses_task_when_llm_is_unavailable(self):
        state = {
            "task": "memory-safe topic",
            "post_to_telegram": False,
            "search_queries": [],
            "search_results": [],
            "scraped_content": [],
            "research_summary": "",
            "twitter_thread": [],
            "final_status": "in_progress",
            "messages": [],
            "error": None,
            "current_agent": "search_agent",
        }

        with patch(
            "agents.search_agent.invoke_prompt",
            side_effect=LLMInvocationError("low memory"),
        ), patch(
            "agents.search_agent.web_search",
            return_value=[{"title": "Result", "url": "https://example.com", "snippet": "x"}],
        ):
            result = search_agent_node(state)

        self.assertEqual(result["search_queries"], ["memory-safe topic"])
        self.assertEqual(len(result["search_results"]), 1)

    def test_research_agent_returns_clear_error_without_search_results(self):
        state = {
            "task": "topic",
            "post_to_telegram": False,
            "search_queries": [],
            "search_results": [],
            "scraped_content": [],
            "research_summary": "",
            "twitter_thread": [],
            "final_status": "in_progress",
            "messages": [],
            "error": None,
            "current_agent": "research_agent",
        }

        result = research_agent_node(state)

        self.assertEqual(result["final_status"], "failed")
        self.assertIn("No search results were returned", result["error"])

    def test_research_agent_builds_fallback_summary(self):
        state = {
            "task": "topic",
            "post_to_telegram": False,
            "search_queries": ["topic"],
            "search_results": [{"title": "Source", "url": "https://example.com"}],
            "scraped_content": [],
            "research_summary": "",
            "twitter_thread": [],
            "final_status": "in_progress",
            "messages": [],
            "error": None,
            "current_agent": "research_agent",
        }

        with patch(
            "agents.research_agent.scrape_with_retry",
            return_value=ScrapeSuccess(
                url="https://example.com",
                content="Revenue grew 42 percent in 2026. Teams adopted faster release cycles.",
                chars=72,
            ),
        ), patch(
            "agents.research_agent.invoke_prompt",
            side_effect=LLMInvocationError("low memory"),
        ):
            result = research_agent_node(state)

        self.assertIn("generated without Ollama", result["research_summary"])
        self.assertEqual(result["current_agent"], "publisher_agent")
        self.assertIsNone(result["error"])

    def test_publisher_agent_builds_fallback_thread(self):
        state = {
            "task": "topic",
            "post_to_telegram": False,
            "search_queries": [],
            "search_results": [],
            "scraped_content": [],
            "research_summary": (
                "KEY FINDINGS\n"
                "- Revenue grew 42 percent year over year.\n"
                "- Teams shipped updates weekly.\n"
                "- Costs fell by 10 percent.\n"
            ),
            "twitter_thread": [],
            "final_status": "in_progress",
            "messages": [],
            "error": None,
            "current_agent": "publisher_agent",
        }

        with patch(
            "agents.publisher_agent.invoke_prompt",
            side_effect=LLMInvocationError("low memory"),
        ):
            result = publisher_agent_node(state)

        self.assertEqual(result["final_status"], "completed")
        self.assertEqual(len(result["twitter_thread"]), 5)
        self.assertTrue(all(len(post) <= MAX_POST_LENGTH for post in result["twitter_thread"]))


class OllamaClientTests(unittest.TestCase):
    def test_invoke_prompt_retries_with_fallback_model_on_low_memory(self):
        def fake_get_llm(model, temperature):
            if model == "llama3.1:8b":
                return SimpleNamespace(
                    invoke=lambda prompt: (_ for _ in ()).throw(
                        RuntimeError("model requires more system memory")
                    )
                )

            return SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="fallback answer"))

        with patch("tools.ollama_client.get_primary_model", return_value="llama3.1:8b"), patch(
            "tools.ollama_client.get_fallback_model",
            return_value="llama3.2:1b",
        ), patch("tools.ollama_client.get_llm", side_effect=fake_get_llm):
            result = invoke_prompt("hello")

        self.assertEqual(result, "fallback answer")
