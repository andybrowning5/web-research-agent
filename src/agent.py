"""Web Research Agent — Deep Agents-powered research agent with web search."""

import json
import os
import sys
from datetime import datetime

import httpx
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_BASE_URL = os.environ.get("BRAVE_BASE_URL", "https://api.search.brave.com")
BRAVE_SEARCH_URL = f"{BRAVE_BASE_URL}/res/v1/web/search"


def send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information. Use this to find current facts, news, documentation, or any topic the user asks about. You can call this multiple times with different queries to get broader coverage."""
    try:
        resp = httpx.get(
            BRAVE_SEARCH_URL,
            params={"q": query, "count": 10},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        log(f"Brave API returned {len(data.get('web', {}).get('results', []))} results for: {query}")
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                f"Title: {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Description: {item.get('description', '')}"
            )
        if not results:
            return "No results found. Try a different search query."
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Search error: {e}"


def research(query: str, message_id: str) -> str:
    """Run the Deep Agent to research a topic."""
    today = datetime.now().strftime("%B %d, %Y")

    agent = create_deep_agent(
        model=init_chat_model("anthropic:claude-sonnet-4-5-20250929"),
        tools=[web_search],
        system_prompt=(
            f"You are Web Research Agent, an expert research assistant. Today is {today}. "
            "Your job is to thoroughly research the user's question using web search. "
            "Strategy:\n"
            "1. Break complex questions into sub-queries and search for each\n"
            "2. Search multiple times with different angles to get comprehensive coverage\n"
            "3. For simple greetings or non-research questions, just respond naturally without searching\n"
            "4. Synthesize all findings into a clear, well-structured briefing with markdown\n"
            "5. Include inline citations [1], [2] etc. and end with a Sources section\n"
            "Be thorough but concise. Prioritize accuracy and recency."
        ),
    )

    final_response = ""
    emitted_tool_calls: set[str] = set()

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        if not isinstance(event, dict) or "messages" not in event:
            continue

        for msg in event["messages"]:
            if (
                getattr(msg, "type", None) == "ai"
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") or tc.get("name", "")
                    if tc_id in emitted_tool_calls:
                        continue
                    emitted_tool_calls.add(tc_id)
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    q = tool_args.get("query", "")
                    desc = f"{tool_name}({q})" if q else tool_name
                    send({
                        "type": "activity",
                        "tool": tool_name,
                        "description": desc,
                        "message_id": message_id,
                    })

        for msg in reversed(event["messages"]):
            if getattr(msg, "type", None) != "ai":
                continue
            msg_content = getattr(msg, "content", "")
            if isinstance(msg_content, str) and msg_content.strip():
                final_response = msg_content
                break
            elif isinstance(msg_content, list):
                for block in msg_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        final_response = block.get("text", "")
                        break
                if final_response:
                    break

    return final_response


def main():
    send({"type": "ready"})
    log("Web Research Agent ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg["type"] == "shutdown":
            log("Shutting down")
            break

        if msg["type"] == "message":
            mid = msg["message_id"]
            query = msg["content"]

            try:
                result = research(query, mid)
                send({
                    "type": "response",
                    "content": result,
                    "message_id": mid,
                    "done": True,
                })
            except Exception as e:
                log(f"Error: {e}")
                send({
                    "type": "error",
                    "error": str(e),
                    "message_id": mid,
                })
                send({
                    "type": "response",
                    "content": f"Something went wrong: {e}",
                    "message_id": mid,
                    "done": True,
                })


if __name__ == "__main__":
    main()
