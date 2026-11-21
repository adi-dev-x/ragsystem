"""Tidy generated answers before they reach the user.

The prompt asks for the right format; this is a safety net for models (small
local ones especially) that still emit Markdown, cite sources that don't exist,
or, for spoken answers, write headings, lists and symbols a voice can't say.
"""

from __future__ import annotations

import re

from app.llm.prompts import AnswerFormat, CitationStyle

_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^([ \t]*)[*+•][ \t]+", re.MULTILINE)
_CODE_RE = re.compile(r"`([^`\n]+)`")
_RULE_RE = re.compile(r"^[ \t]*([-*_])[ \t]*(\1[ \t]*){2,}$", re.MULTILINE)
_CITATION_RE = re.compile(r"([ \t]*)\[(\d+(?:\s*,\s*\d+)*)\]")
_SPACE_AFTER_CITATION_RE = re.compile(r"\][ \t]+([.,;:!?])")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def strip_markdown(text: str) -> str:
    text = _RULE_RE.sub("", text)
    text = _HEADING_RE.sub(lambda m: m.group(1).rstrip(":") + ":", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    text = _BULLET_RE.sub(r"\1- ", text)
    return text


def normalise_citations(text: str, source_count: int) -> str:
    """Rewrite [1, 3] as [1][3] and drop numbers that don't match a source."""

    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(2))]
        if any(n > 99 for n in numbers):
            return match.group(0)  # e.g. "[2024]" is a year, not a citation
        valid = [n for n in dict.fromkeys(numbers) if 1 <= n <= source_count]
        if not valid:
            return ""  # drop the citation and the space before it
        return match.group(1) + "".join(f"[{n}]" for n in valid)

    return _CITATION_RE.sub(replace, text)


# --------------------------------------------------------------------------
# Spoken answers
# --------------------------------------------------------------------------

# [1], [1][2], [1, 3], [2-4] and written citations such as [file.pdf, p. 12]
_ANY_NUMERIC_CITATION_RE = re.compile(r"[ \t]*\[\d{1,2}(?:\s*[,\u2013-]\s*\d{1,2})*\]")
_DOCUMENT_CITATION_RE = re.compile(r"[ \t]*\[[^\]\n]*(?:\.pdf|\bpp?\.\s*\d)[^\]\n]*\]", re.IGNORECASE)
_SOURCE_PAREN_RE = re.compile(r"[ \t]*\((?:see\s+)?(?:source|sources|document|page|pp?\.)\s*[^)]{0,40}\)", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+\u2022]|\d{1,2}[.)])[ \t]+(.*)$")
_LABEL_LINE_RE = re.compile(r"^[A-Z][^.!?\n]{0,48}:$")
_INLINE_LABEL_RE = re.compile(r"^([A-Z][A-Za-z0-9 ,'/&-]{1,40}):[ \t]+(?=\S)")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF\uFE0F\u200d]"
)
_OPENER_RE = re.compile(
    r"^(?:(?:certainly|sure|of course|absolutely|great question|good question)[!,.]*\s+"
    r"(?:i'?d be (?:happy|glad) to help(?: with that)?[!,.]*\s*)?)?"
    r"(?:(?:based on|according to) (?:the )?(?:available|provided|retrieved|given)? ?"
    r"(?:information|documents?|context|excerpts?|sources?)(?: (?:provided|available))?[,:]\s*)?",
    re.IGNORECASE,
)
_SPOKEN_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\be\.\s?g\.,?(?=\s)", re.IGNORECASE), "for example,"),
    (re.compile(r"\bi\.\s?e\.,?(?=\s)", re.IGNORECASE), "that is,"),
    (re.compile(r",?\s+etc\.(?=\s|$)"), " and so on."),
    (re.compile(r"\s*(?:->|\u2192|=>)\s*"), " to "),
    (re.compile(r"\s+&\s+"), " and "),
    (re.compile(r"[*#_`|~^<>{}\[\]]"), ""),
)


def _sentence(text: str) -> str:
    text = text.strip().rstrip(";,")
    if text and text[-1] not in ".!?":
        text += "."
    return text[:1].upper() + text[1:] if text else text


def _flatten_lists_and_labels(text: str) -> str:
    """Turn headings, labels and list items into running sentences."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        sentences: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            item = _LIST_ITEM_RE.match(line)
            if item:
                line = item.group(1).strip()
                if line:
                    sentences.append(_sentence(line))
                continue
            if _LABEL_LINE_RE.match(line) and len(line.split()) <= 6:
                continue  # a heading such as "Trust controls:"
            label = _INLINE_LABEL_RE.match(line)
            if label and len(label.group(1).split()) <= 5:
                line = line[label.end():]  # "Language and input: Users can..." -> "Users can..."
            if sentences and sentences[-1].endswith(":"):
                sentences[-1] = sentences[-1][:-1] + "."
            sentences.append(line[:1].upper() + line[1:])
        if sentences:
            if sentences[-1].endswith(":"):
                sentences[-1] = sentences[-1][:-1] + "."
            paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


def to_speech(text: str) -> str:
    """Make an answer safe to pass straight to a text-to-speech engine."""
    text = strip_markdown(text)
    text = _DOCUMENT_CITATION_RE.sub("", text)
    text = _ANY_NUMERIC_CITATION_RE.sub("", text)
    text = _SOURCE_PAREN_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _flatten_lists_and_labels(text)
    for pattern, replacement in _SPOKEN_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = text.lstrip()
    opener = _OPENER_RE.match(text)
    if opener and opener.end():
        text = text[opener.end():]
        text = text[:1].upper() + text[1:]
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.!?])\1+", r"\1", text)
    text = re.sub(r"\.\s*,", ".", text)
    return _BLANK_LINES_RE.sub("\n\n", _TRAILING_SPACE_RE.sub("", text)).strip()


# --------------------------------------------------------------------------


def clean_answer(text: str, *, answer_format: AnswerFormat, citation_style: CitationStyle, source_count: int) -> str:
    if answer_format == "spoken":
        return to_speech(text)
    if answer_format == "plain":
        text = strip_markdown(text)
    if citation_style == "numbered":
        text = normalise_citations(text, source_count)
        text = _SPACE_AFTER_CITATION_RE.sub(r"]\1", text)  # "word [1] ." -> "word [1]."
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()
