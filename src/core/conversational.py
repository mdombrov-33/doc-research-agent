"""Pre-graph short-circuit for greetings and meta-questions.

Conservative by construction: only a fixed set of normalized greetings/thanks and a few
help phrases match. Anything carrying real content words falls through to the retrieval
graph, so a document question is never misrouted here — the worst case is the same hard
abstention as before, never a wrong answer.
"""

import re

CAPABILITIES_RESPONSE = (
    "I'm a document research assistant. Upload documents, then ask questions about them — "
    "I answer from their contents and cite the passages I use. If the documents don't cover "
    "a question I can fall back to a web search, and when nothing supports an answer I say so "
    "rather than guess."
)

# Whole-message greetings and thanks that carry no question to research (normalized form:
# lowercased, punctuation stripped).
_GREETINGS = frozenset(
    {
        "hi",
        "hii",
        "hello",
        "hey",
        "heya",
        "hiya",
        "yo",
        "howdy",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "thank you very much",
        "thanks a lot",
        "many thanks",
        "ty",
        "thx",
        "cheers",
        "ok thanks",
        "okay thanks",
        "bye",
        "goodbye",
        "see you",
        "see ya",
    }
)

# Whole-message requests to describe what the assistant does.
_HELP = frozenset(
    {
        "help",
        "what can you do",
        "what do you do",
        "who are you",
        "what are you",
        "how do you work",
        "how does this work",
        "what is this",
        "what can i ask",
        "what can you help with",
        "what can you help me with",
    }
)


def _normalize(question: str) -> str:
    text = re.sub(r"[^\w\s]", "", question.lower())
    return re.sub(r"\s+", " ", text).strip()


def match(question: str) -> str | None:
    """Return the fixed capabilities response for a greeting/meta message, else None."""
    normalized = _normalize(question)
    if normalized in _GREETINGS or normalized in _HELP:
        return CAPABILITIES_RESPONSE
    return None
