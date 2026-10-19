"""Streamlit chat UI for the NBA Guide agent (local only).

    streamlit run app.py

The agent is built once per process (st.cache_resource). Each session may ask APP_MAX_QUESTIONS
questions (default 30) to protect the LLM key; Clear chat empties the history but not that count.
Errors become friendly messages in the chat; key values are never shown.
"""

from __future__ import annotations

import logging
from itertools import pairwise
from typing import Any

import streamlit as st
from langchain_core.runnables import Runnable

import agent
import config

logger = logging.getLogger(__name__)

PAGE_TITLE = "NBA Guide Q&A"
INPUT_PLACEHOLDER = "Ask about the 2026-27 NBA Guide..."
SPINNER_TEXT = "Searching the guide and the stats tables... answers can take up to a minute."
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("Narrative", "What is the record for most points by one player in a single regular-season game?"),
    ("Numeric", "How many NBA Finals has Boston won?"),
    ("Mixed", "Who won the 2026 NBA Finals, in how many games, and who was the Finals MVP?"),
    ("Season list", "Who was on the 1996-97 All-NBA first team?"),
    ("Not in the guide", "Who won the 2026 Stanley Cup?"),
)


def pdf_pages(pages: list[int]) -> str:
    return ", ".join(f"PDF p. {p}" for p in pages)


@st.cache_resource(show_spinner=False)
def get_agent() -> Runnable[Any, Any]:
    """Builds the agent once per process; a ConfigError is not cached, so fixing .env and rerunning works."""
    return agent.build_agent()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("questions_asked", 0)
    st.session_state.setdefault("pending", None)


def max_questions() -> int:
    try:
        return config.get_app_max_questions()
    except ValueError as e:
        logger.warning("Invalid APP_MAX_QUESTIONS (%s); using the default.", e)
        return config.DEFAULT_APP_MAX_QUESTIONS


def cap_reached() -> bool:
    return int(st.session_state.questions_asked) >= max_questions()


def history_pairs() -> list[tuple[str, str]]:
    """(question, answer) pairs of answered turns, oldest first; ask() keeps the last HISTORY_TURNS."""
    pairs: list[tuple[str, str]] = []
    messages = st.session_state.messages
    for user, reply in pairwise(messages):
        if user["role"] == "user" and reply["role"] == "assistant" and not reply.get("error"):
            pairs.append((user["content"], reply["content"]))
    return pairs


def step_dicts(steps: list[agent.ToolStep]) -> list[dict[str, Any]]:
    return [{"name": s.name, "args": dict(s.args), "pages": list(s.pages), "warning": s.warning} for s in steps]


def submit(question: str) -> None:
    """Queues a question; the next run disables the input and answers it."""
    question = question.strip()
    if not question or st.session_state.pending is not None or cap_reached():
        return
    st.session_state.pending = question
    st.rerun()


def answer(question: str) -> dict[str, Any]:
    """Runs the agent and returns an assistant message dict (answer or friendly error)."""
    try:
        graph = get_agent()
        result = agent.ask(graph, question, history=history_pairs())
    except agent.QuotaExceededError as e:
        return {"role": "assistant", "content": f"The model's rate limit or quota was reached. {e}",
                "error": "quota"}
    except agent.TooManyStepsError as e:
        return {"role": "assistant", "content": str(e), "error": "steps", "steps": step_dicts(e.steps)}
    except agent.LLMTimeoutError as e:
        return {"role": "assistant", "content": str(e), "error": "timeout"}
    except agent.ConfigError as e:
        return {"role": "assistant", "content": f"Configuration problem: {e}", "error": "config"}
    except (agent.AgentError, ValueError) as e:
        return {"role": "assistant", "content": f"Sorry, I couldn't answer that: {e}", "error": "agent"}
    except Exception as e:  # UI boundary: keep the chat usable whatever the provider raises
        logger.exception("Unexpected error while answering")
        return {"role": "assistant", "content": f"Something went wrong ({type(e).__name__}). Please try again.",
                "error": "unexpected"}
    return {
        "role": "assistant",
        "content": result.answer,
        "steps": step_dicts(result.steps),
        "verified_pages": result.verified_pages,
        "unverified_pages": result.unverified_pages,
        "model_calls": result.model_calls,
        "seconds": result.seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def render_steps(steps: list[dict[str, Any]]) -> None:
    for i, step in enumerate(steps, start=1):
        args = step.get("args", {})
        if "sql" in args:
            st.markdown(f"**{i}. run_sql**")
            st.code(args["sql"], language="sql")
        elif "query" in args:
            st.markdown(f"**{i}. search_guide:** {args['query']}")
        else:
            st.markdown(f"**{i}. {step.get('name', 'tool')}:** {args}")
        if step.get("warning"):
            st.warning(f"Step {i}: {step['warning']}")


def render_assistant(message: dict[str, Any]) -> None:
    error = message.get("error")
    if error in ("timeout",):
        st.warning(message["content"])
    elif error:
        st.error(message["content"])
    else:
        st.markdown(message["content"])
        st.caption(f"Answered in {message.get('seconds', 0.0):.1f}s")

    steps = message.get("steps") or []
    if error and not steps:
        return
    with st.expander("How I got this"):
        if steps:
            render_steps(steps)
        else:
            st.markdown("No tools were used.")
        if error:
            return
        verified = message.get("verified_pages") or []
        st.markdown(f"**Verified pages:** {pdf_pages(verified) if verified else 'none'}")
        unverified = message.get("unverified_pages") or []
        if unverified:
            st.warning(f"Cited but not found in any tool output: {pdf_pages(unverified)}")
        st.caption(
            f"Model calls: {message.get('model_calls', 0)} · Latency: {message.get('seconds', 0.0):.1f}s · "
            f"Tokens: {message.get('input_tokens', 0)} in / {message.get('output_tokens', 0)} out"
        )


def render_sidebar() -> None:
    busy = st.session_state.pending is not None
    with st.sidebar:
        st.header("About")
        st.markdown(f"**Model:** {agent.llm_label()}")
        st.caption(f"Questions this session: {st.session_state.questions_asked} / {max_questions()}")
        st.subheader("Try an example")
        for label, question in EXAMPLES:
            if st.button(f"{label}: {question}", key=f"example_{label}", disabled=busy or cap_reached(),
                         width="stretch"):
                submit(question)
        if st.button("Clear chat", key="clear_chat", disabled=busy, width="stretch"):
            st.session_state.messages = []
            st.rerun()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE)
    st.title(PAGE_TITLE)
    init_state()
    render_sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message)

    pending = st.session_state.pending
    if cap_reached() and pending is None:
        st.info(f"This session reached its limit of {max_questions()} questions. "
                "Restart the app to ask more (APP_MAX_QUESTIONS sets the limit).")
    prompt = st.chat_input(INPUT_PLACEHOLDER, disabled=pending is not None or cap_reached())
    if prompt:
        submit(prompt)

    if pending is not None:
        with st.chat_message("user"):
            st.markdown(pending)
        with st.chat_message("assistant"), st.spinner(SPINNER_TEXT):
            reply = answer(pending)
        st.session_state.messages += [{"role": "user", "content": pending}, reply]
        st.session_state.questions_asked += 1
        st.session_state.pending = None
        st.rerun()


main()
