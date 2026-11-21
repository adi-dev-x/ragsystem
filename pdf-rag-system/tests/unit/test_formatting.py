"""Answer cleanup and the format/citation prompt variants."""

import pytest

from app.llm.formatting import clean_answer, normalise_citations, strip_markdown, to_speech
from app.llm.prompts import NOT_FOUND_ANSWER, SPOKEN_NOT_FOUND_ANSWER, build_system_prompt, build_user_prompt

RAW = """## Overview

The BambooLoops AI assistant is a **multilingual, voice-enabled** help layer [1].

**What it does**
* Helps users discover services [1][2] .
* Supports Malayalam and English at launch [1, 3].

---

Uses `retrieval-grounded` answers [7]. Founded in [2024]."""


def test_plain_answer_has_no_markdown_and_valid_citations() -> None:
    cleaned = clean_answer(RAW, answer_format="plain", citation_style="numbered", source_count=3)
    assert cleaned == (
        "Overview:\n\n"
        "The BambooLoops AI assistant is a multilingual, voice-enabled help layer [1].\n\n"
        "What it does\n"
        "- Helps users discover services [1][2].\n"
        "- Supports Malayalam and English at launch [1][3].\n\n"
        "Uses retrieval-grounded answers. Founded in [2024]."
    )


def test_markdown_format_keeps_markdown() -> None:
    cleaned = clean_answer("**Bold** [1]", answer_format="markdown", citation_style="numbered", source_count=1)
    assert cleaned == "**Bold** [1]"


def test_document_citations_are_left_untouched() -> None:
    text = "Two servers [a.pdf, p. 1]."
    assert clean_answer(text, answer_format="plain", citation_style="document", source_count=1) == text


def test_strip_markdown_keeps_hyphen_lists_and_arithmetic() -> None:
    assert strip_markdown("- item one\n- item two") == "- item one\n- item two"
    assert strip_markdown("2 * 3 * 4 = 24") == "2 * 3 * 4 = 24"


def test_normalise_citations_drops_unknown_numbers() -> None:
    assert normalise_citations("Fact [2]. Other [9]. Both [1, 2, 2].", source_count=2) == "Fact [2]. Other. Both [1][2]."


def test_prompt_variants() -> None:
    plain_numbered = build_system_prompt("plain", "numbered")
    assert "Do not use Markdown" in plain_numbered and "[1]" in plain_numbered
    assert "[architecture.pdf, p. 12]" not in plain_numbered

    markdown_document = build_system_prompt("markdown", "document")
    assert "Markdown" in markdown_document and "[architecture.pdf, p. 12]" in markdown_document

    for prompt in (plain_numbered, markdown_document):
        assert NOT_FOUND_ANSWER in prompt and "outside knowledge" in prompt and "contradict" in prompt


# --------------------------------------------------------------------------
# Spoken answers
# --------------------------------------------------------------------------

DOC_STYLE = """The BambooLoops AI assistant is a voice-enabled access layer across the platform [2][3][1].

Language and input: Users can type or speak in Malayalam and English at launch [1].

**Trust controls:**
- visible transcripts
- confirmation before consequential actions [1, 2]
- escalation to a human expert [BambooLoops.pdf, pp. 10-16]"""


def test_to_speech_removes_citations_headings_and_lists() -> None:
    assert to_speech(DOC_STYLE) == (
        "The BambooLoops AI assistant is a voice-enabled access layer across the platform.\n\n"
        "Users can type or speak in Malayalam and English at launch.\n\n"
        "Visible transcripts. Confirmation before consequential actions. Escalation to a human expert."
    )


def test_conversational_text_passes_through_unchanged() -> None:
    natural = (
        "You can simply ask questions in Malayalam or English, and the assistant will help you find services.\n\n"
        "Currently, AI-powered FAQs are part of the MVP. More advanced retrieval is planned for a later stage."
    )
    assert to_speech(natural) == natural


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Certainly! The MVP includes AI FAQs.", "The MVP includes AI FAQs."),
        ("Sure, I'd be happy to help! It supports Malayalam.", "It supports Malayalam."),
        ("Based on the available information, it supports Malayalam.", "It supports Malayalam."),
        ("Based on the provided documents: it is a pilot.", "It is a pilot."),
        ("Surely this works.", "Surely this works."),  # not an opener
    ],
)
def test_filler_openers_are_removed(raw: str, expected: str) -> None:
    assert to_speech(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Crops, e.g. bamboo, are listed.", "Crops, for example, bamboo, are listed."),
        ("Farmers & buyers connect 😀.", "Farmers and buyers connect."),
        ("Lot -> batch -> order.", "Lot to batch to order."),
        ("Roles include farmers, traders, etc.", "Roles include farmers, traders and so on."),
        ("Phase 2 (see page 15) adds payments.", "Phase 2 adds payments."),
        ("It launched in [2024] for *all* users.", "It launched in 2024 for all users."),
    ],
)
def test_symbols_and_abbreviations_are_spoken_friendly(raw: str, expected: str) -> None:
    assert to_speech(raw) == expected


def test_spoken_format_ignores_citation_style() -> None:
    text = "Two servers share one database [1]."
    for style in ("numbered", "document"):
        assert clean_answer(text, answer_format="spoken", citation_style=style, source_count=1) == "Two servers share one database."


def test_spoken_prompt() -> None:
    prompt = build_system_prompt("spoken", "numbered")
    for phrase in ("text-to-speech", "only source of facts", "Do not invent", "Never mention sources",
                   "No headings", "bullet points", "Certainly", "disagree", "planned, proposed or phased"):
        assert phrase in prompt, phrase
    assert SPOKEN_NOT_FOUND_ANSWER in prompt
    assert "[1]" not in prompt and "cite" not in prompt.lower().replace("excerpt", "")
    assert "with citations" not in build_user_prompt("q", "ctx", "spoken")
