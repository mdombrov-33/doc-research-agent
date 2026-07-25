import pytest

from src.core.conversational import CAPABILITIES_RESPONSE, match


@pytest.mark.parametrize(
    "question",
    ["hi", "Hello!", "  HEY  ", "thanks", "Thank you very much.", "what can you do?", "help"],
)
def test_greetings_and_meta_questions_get_the_fixed_reply(question):
    assert match(question) == CAPABILITIES_RESPONSE


@pytest.mark.parametrize(
    "question",
    [
        "hi, what does the contract say about termination?",
        "what is the refund policy?",
        "thanks for the report, what were the Q3 numbers?",
        "who is the CEO mentioned in the document?",
    ],
)
def test_content_bearing_questions_fall_through(question):
    assert match(question) is None
