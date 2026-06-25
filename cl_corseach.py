# cl_corsearch.py , loction - vendor_pipeline\common_law (folder)
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import threading
from concurrent.futures import ThreadPoolExecutor

import fitz
import instructor
from instructor.exceptions import (
    IncompleteOutputException,
    InstructorRetryException,
    ResponseParsingError,
)
import pdfplumber
from openai import AzureOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError


FOOTER_PAGE_RE = re.compile(r"\bPage\s+(\d+)\b", re.IGNORECASE)
DOC_NO_RE = re.compile(r"^CML-\d+[A-Z]?$", re.IGNORECASE)
CML_RE = re.compile(r"^CML-(\d+[A-Z]?)$", re.IGNORECASE)
PUBLICATION_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2},\s+\d{4}(?:\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))?\b",
    re.IGNORECASE,
)
SUMMARY_TITLE = "C O M M O N L A W S U M M A R Y"
SEARCH_RESULTS_TITLE = "C O M M O N L A W S E A R C H R E S U L T S"
GLOBAL_NEW_PRODUCTS_TITLE = "G L O B A L N E W P R O D U C T S S E A R C H R E S U L T S"
_LLM_TIMEOUT_SECONDS = 30.0
HIGH_COMMERCIAL_PHRASES = (
    "announced the launch",
    "introduced",
    "new product",
    "brand",
    "available in stores",
    "available at dispensaries",
    "launch of",
    "product line",
    "flavor",
    "hard seltzer",
    "tea bags",
    "ice cream",
    "restaurant",
    "cocktail",
    "sold under",
    "company",
    "beverage",
    "consumer product",
    "distributed by",
    "manufactured by",
)
VERY_STRONG_OWNER_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'. -]{2,80}?)\s+"
    r"(?:announced|launched|partnered with|introduced|unveiled|released)\b"
    r"|\bfrom\s+([A-Z][A-Za-z0-9&'. -]{2,80}?)\b"
    r"|\b(?:created|manufactured|distributed)\s+by\s+([A-Z][A-Za-z0-9&'. -]{2,80}?)\b"
)
COMMERCIAL_NOUNS = (
    "beverage",
    "tea",
    "coffee",
    "cocktail",
    "beer",
    "wine",
    "restaurant",
    "smoothie",
    "smoothies",
    "ice cream",
    "cannabis",
    "flower jars",
    "vape",
    "seltzer",
    "food",
    "drink",
    "dessert",
    "snack",
    "hard seltzer",
    "menu",
    "brewery",
    "dining",
    "pre-workout",
)
LOW_VALUE_PHRASES = (
    "event listing",
    "music schedule",
    "editorial discussion",
    "generic article mention",
    "band listing",
    "festival listing",
    "social discussion",
    "calendar",
    "concert schedule",
)
OWNER_PATTERN_RE_LIST = (
    re.compile(r"\b([A-Z][A-Za-z0-9&'. -]{2,90}?)\s+announced\b"),
    re.compile(r"\b([A-Z][A-Za-z0-9&'. -]{2,90}?)\s+launched\b"),
    re.compile(r"\b([A-Z][A-Za-z0-9&'. -]{2,90}?)\s+partnered with\b"),
    re.compile(r"\blaunch of\s+.{0,80}?\s+by\s+([A-Z][A-Za-z0-9&'. -]{2,90}?)\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+([A-Z][A-Za-z0-9&'. -]{2,90}?)\b"),
    re.compile(r"\bmanufactured by\s+([A-Z][A-Za-z0-9&'. -]{2,90}?)\b", re.IGNORECASE),
    re.compile(r"\bcreated by\s+([A-Z][A-Za-z0-9&'. -]{2,90}?)\b", re.IGNORECASE),
    re.compile(r"\bavailable at\s+([A-Z][A-Za-z0-9&'. -]{2,90}?)\b", re.IGNORECASE),
)
PUBLICATION_OWNER_TERMS = (
    "newspaper",
    "magazine",
    "journal",
    "tribune",
    "herald",
    "times",
    "gazette",
    "daily news",
    "usa today",
    "yahoo finance",
    "inc.com",
    "web edition",
    "publication",
    "byline",
    "section",
)
GENERIC_GOODS_VALUES = {"product", "products", "service", "services", "goods", "mark"}
OWNER_STOPWORDS = {
    "new",
    "number",
    "france",
    "usa",
    "united states",
    "english",
    "copyright",
    "source",
    "content",
}
TRACK_A_DESCRIPTION_TERMS = (
    "business name information",
    "dun's market identifiers",
    "duns market identifiers",
    "corporate registries",
    "corporate registry",
    "state trademark registry",
    "federal trademark registry",
    "trademark registry",
)
TRACK_B_DESCRIPTION_TERMS = ("transcript", "newswire")
TRACK_C_DESCRIPTION_TERMS = (
    "newspaper",
    "magazine",
    "aggregate news",
    "web publication",
    "internet",
)
TRANSCRIPT_OWNER_RE = re.compile(
    r"^(?:Q[1-4]|H[1-2]|\d{4})\s+\d{4}\s+(.+?)\s+"
    r"(?:Earnings Call|Corporate Access|Corporate Presentation)\b",
    re.IGNORECASE,
)
NEWSWIRE_SOURCE_RE = re.compile(
    r"SOURCE\s+([A-Za-z0-9\s,.\-+&']+?)(?=\s+LANGUAGE\b|$)",
    re.IGNORECASE,
)
GEOGRAPHIC_INDICATOR_RE = re.compile(
    r"\b(?:STATE|COUNTRY|REGION):\s*([^:]+?)(?=\s+(?:LANGUAGE|SUBJECT|COUNTRY|REGION|STATE):|$)",
    re.IGNORECASE,
)
LLM_SYSTEM_PROMPT = """You are a trademark clearance analyst performing commercial-use attribution.
PRIMARY OBJECTIVE
Determine:
1. owner_name
2. goods_services
3. nice_class
for the target mark using only the supplied evidence.
OWNER IDENTIFICATION RULES-
The owner_name must represent the entity most likely responsible for offering, manufacturing, distributing, operating, licensing, marketing, or selling the goods or services associated with the mark.
Never treat the following as the owner unless they are explicitly identified as the commercial source of the mark:
newspapers, magazines, publishers, journalists, news agencies, web portals, editorial organizations, article authors

OWNER PRIORITY ORDER-
When multiple entities appear, rank them in the following order:
1. Manufacturer
2. Brand Owner
3. Distributor
4. Service Provider
5. Operator
6. Retailer
Prefer the highest-ranked entity supported by the evidence.
GOODS AND SERVICES RULES-
Return a concise commercial description of the goods or services associated with the mark.
Do not return:
* article topics
* publication categories
* editorial descriptions
* generic business descriptions unrelated to the mark

NICE CLASS RULES-Assign a Nice Class only from the identified goods or services.
The class must be an integer from 1 to 45. Never derive Nice Class from:
* company type
* publication type
* article category
* owner name

ATTRIBUTION RULES-
Use the strongest commercially relevant evidence available.
When direct ownership evidence is unavailable, select the most commercially plausible owner from the supplied evidence and commercial context.
When direct goods or services evidence is unavailable, derive the most commercially reasonable goods or services description supported by the evidence.
Prefer evidence-supported attribution over empty fields.
Do not invent entities, products, services, or facts that are not present in the supplied evidence.
"""


class CorsearchAnalysis(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    owner_name: str | None = None
    goods_services: str | None = None
    nice_class: int | None = None


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_spaced_heading(value: str) -> str:
    return normalize_cell(value).replace("  ", " ")


def is_bold_font(font_name: str) -> bool:
    return "bold" in font_name.lower() or "black" in font_name.lower()


def page_lines(page: fitz.Page) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [
                span
                for span in line.get("spans", [])
                if normalize_cell(span.get("text", ""))
            ]
            if not spans:
                continue
            text = normalize_cell(" ".join(normalize_cell(span.get("text", "")) for span in spans))
            lines.append(
                {
                    "text": text,
                    "bbox": tuple(float(value) for value in line["bbox"]),
                    "is_bold": any(is_bold_font(span.get("font", "")) for span in spans),
                    "max_size": max(float(span.get("size", 0.0)) for span in spans),
                }
            )
    return sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def line_same_row(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_mid = (float(first["bbox"][1]) + float(first["bbox"][3])) / 2
    second_mid = (float(second["bbox"][1]) + float(second["bbox"][3])) / 2
    return abs(first_mid - second_mid) <= max(float(first["max_size"]), float(second["max_size"])) * 0.7


def footer_page_number(page: fitz.Page) -> int | None:
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    footer_rect = fitz.Rect(
        page_width * 0.65,
        page_height * 0.88,
        page_width,
        page_height,
    )
    footer_text = page.get_text("text", clip=footer_rect)
    match = FOOTER_PAGE_RE.search(footer_text)
    if match:
        return int(match.group(1))

    match = FOOTER_PAGE_RE.search(page.get_text("text"))
    return int(match.group(1)) if match else None


def calculate_page_range(pdf_path: str, context: Any = None) -> tuple[str, str]:
    try:
        try:
            from ..vender_detection_state import extract_corsearch_common_law_pdf
        except (ImportError, ValueError):
            from vender_detection_state import extract_corsearch_common_law_pdf
    except ImportError:
        # Fallback if it's imported from somewhere else or sys.path is different
        import importlib
        try:
            vds = importlib.import_module("vender_detection_state")
            extract_corsearch_common_law_pdf = vds.extract_corsearch_common_law_pdf
        except ImportError:
            raise

    doc = context.get_fitz_doc() if context else fitz.open(pdf_path)
    try:
        page_range = extract_corsearch_common_law_pdf(doc)
    finally:
        if not context:
            doc.close()

    return (
        page_range.get("Web_corsearch_start_page", "error"),
        page_range.get("Web_corsearch_end_page", "error"),
    )


def locate_start_page_index(doc: fitz.Document, start_page: int) -> int | None:
    for page_index, page in enumerate(doc):
        if footer_page_number(page) == start_page:
            return page_index
    return None


def is_summary_page(page: fitz.Page) -> bool:
    text = normalize_spaced_heading(page.get_text("text"))
    return SUMMARY_TITLE in text and "COMMON LAW TRADEMARK REFERENCE" in text


def is_search_results_page(page: fitz.Page) -> bool:
    return SEARCH_RESULTS_TITLE in normalize_spaced_heading(page.get_text("text"))


def valid_record(reference: str, doc_no: str, page: str) -> bool:
    populated = sum(bool(value) for value in (reference, doc_no, page))
    return bool(
        populated >= 2
        and reference
        and DOC_NO_RE.fullmatch(doc_no)
        and page.isdigit()
    )


def record_from_cells(cells: list[Any]) -> dict[str, str] | None:
    normalized = [normalize_cell(cell) for cell in cells]
    if len(normalized) < 3:
        return None

    page = normalized[-1]
    doc_no = normalized[-2]
    reference = normalize_cell(" ".join(cell for cell in normalized[:-2] if cell))

    if not valid_record(reference, doc_no, page):
        return None

    return {
        "COMMON LAW TRADEMARK REFERENCE": reference,
        "Doc No.": doc_no,
        "Page": page,
    }


def extract_table_records(page: pdfplumber.page.Page) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    tables = page.extract_tables() or []
    for table in tables:
        for row in table:
            record = record_from_cells(row)
            if record:
                records.append(record)
    return records


def words_to_rows(words: list[dict[str, Any]], y_tolerance: float = 3.0) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        if not rows:
            rows.append([word])
            continue

        current_top = sum(float(item["top"]) for item in rows[-1]) / len(rows[-1])
        if abs(float(word["top"]) - current_top) <= y_tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])

    return [sorted(row, key=lambda item: float(item["x0"])) for row in rows]


def extract_position_records(page: pdfplumber.page.Page) -> list[dict[str, str]]:
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    records: list[dict[str, str]] = []

    for row in words_to_rows(words):
        doc_words = [word for word in row if DOC_NO_RE.fullmatch(normalize_cell(word["text"]))]
        page_words = [word for word in row if normalize_cell(word["text"]).isdigit() and float(word["x0"]) >= 470]
        if not doc_words or not page_words:
            continue

        doc_word = doc_words[0]
        page_word = page_words[-1]
        reference_words = [word for word in row if float(word["x1"]) < float(doc_word["x0"]) - 5]
        reference = normalize_cell(" ".join(normalize_cell(word["text"]) for word in reference_words))
        doc_no = normalize_cell(doc_word["text"])
        page_number = normalize_cell(page_word["text"])

        if not valid_record(reference, doc_no, page_number):
            continue

        records.append(
            {
                "COMMON LAW TRADEMARK REFERENCE": reference,
                "Doc No.": doc_no,
                "Page": page_number,
            }
        )

    return records


def dedupe_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            record["COMMON LAW TRADEMARK REFERENCE"],
            record["Doc No."],
            record["Page"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def build_page_index_by_footer(
    doc: fitz.Document,
    start_page_index: int,
    start_page: int,
    last_allowed_page: int,
) -> dict[int, int]:
    page_index_by_footer: dict[int, int] = {}
    for page_index in range(start_page_index, doc.page_count):
        logical_page_number = footer_page_number(doc[page_index])
        if logical_page_number is None:
            continue
        if logical_page_number > last_allowed_page:
            break
        if logical_page_number >= start_page:
            page_index_by_footer[logical_page_number] = page_index
    return page_index_by_footer


def locate_search_results_start_index(
    doc: fitz.Document,
    start_page_index: int,
    last_allowed_page: int,
) -> int | None:
    for page_index in range(start_page_index, doc.page_count):
        logical_page_number = footer_page_number(doc[page_index])
        if logical_page_number is None:
            continue
        if logical_page_number > last_allowed_page:
            break
        if is_search_results_page(doc[page_index]):
            return page_index
    return None


def line_value_after_label(lines: list[dict[str, Any]], label_line: dict[str, Any]) -> str:
    same_row_values = [
        line
        for line in lines
        if line is not label_line
        and line_same_row(label_line, line)
        and float(line["bbox"][0]) > float(label_line["bbox"][2])
    ]
    return normalize_cell(" ".join(line["text"] for line in sorted(same_row_values, key=lambda item: item["bbox"][0])))


def find_label_line(lines: list[dict[str, Any]], labels: tuple[str, ...], min_y: float = 0.0) -> dict[str, Any] | None:
    for line in lines:
        if float(line["bbox"][1]) < min_y:
            continue
        text = normalize_cell(line["text"])
        if any(text == label or text.startswith(label + " ") for label in labels):
            return line
    return None


def is_publication_date_label(line: dict[str, Any]) -> bool:
    return bool(
        line["is_bold"]
        and float(line["bbox"][0]) < 80
        and normalize_cell(line["text"]).startswith("Publication Date:")
    )


def find_publication_date_line(lines: list[dict[str, Any]], min_y: float = 0.0) -> dict[str, Any] | None:
    for line in lines:
        if float(line["bbox"][1]) < min_y:
            continue
        if is_publication_date_label(line):
            return line
    return None


def collect_lines_between(
    lines: list[dict[str, Any]],
    start_line: dict[str, Any],
    stop_line: dict[str, Any] | None,
    left_min: float = 0.0,
) -> list[str]:
    start_bottom = float(start_line["bbox"][3])
    stop_top = float(stop_line["bbox"][1]) if stop_line is not None else float("inf")
    values: list[str] = []
    inline_value = line_value_after_label(lines, start_line)
    if inline_value:
        values.append(inline_value)

    for line in lines:
        y0 = float(line["bbox"][1])
        if y0 <= start_bottom:
            continue
        if stop_line is not None and (y0 >= stop_top or line_same_row(line, stop_line)):
            continue
        if float(line["bbox"][0]) < left_min:
            continue
        values.append(line["text"])

    return values


def first_publication_date(lines: list[dict[str, Any]], min_y: float) -> str:
    publication_date_line = find_publication_date_line(lines, min_y)
    if publication_date_line is not None:
        same_row_value = line_value_after_label(lines, publication_date_line)
        if same_row_value:
            return same_row_value

    for line in lines:
        if float(line["bbox"][1]) < min_y:
            continue
        match = re.search(r"\bPublication Date:\s*(.+)$", line["text"], flags=re.IGNORECASE)
        if match:
            return normalize_cell(match.group(1))
        match = PUBLICATION_DATE_RE.search(line["text"])
        if match and "LOAD-DATE" not in line["text"].upper():
            return normalize_cell(match.group(0))
    return ""


def collect_content_page_lines(
    lines: list[dict[str, Any]],
    min_y: float,
    publication_date_line: dict[str, Any] | None,
) -> str:
    stop_top = float(publication_date_line["bbox"][1]) if publication_date_line is not None else float("inf")
    values: list[str] = []
    for line in lines:
        y0 = float(line["bbox"][1])
        if y0 < min_y or y0 >= 730 or y0 >= stop_top:
            continue
        if publication_date_line is not None and line_same_row(line, publication_date_line):
            continue
        if float(line["bbox"][0]) < 95:
            continue
        values.append(line["text"])
    return "\n".join(normalize_cell(value) for value in values if normalize_cell(value))


def extract_full_content(
    doc: fitz.Document,
    start_page_index: int,
    content_line: dict[str, Any] | None,
    next_record_page_index: int | None,
    last_allowed_page_index: int,
) -> tuple[str, str]:
    if content_line is None:
        return "", ""

    content_parts: list[str] = []
    date = ""
    end_page_index = min(
        last_allowed_page_index,
        next_record_page_index - 1 if next_record_page_index is not None else last_allowed_page_index,
    )

    start_lines = page_lines(doc[start_page_index])
    publication_date_line = find_publication_date_line(start_lines, float(content_line["bbox"][3]))
    inline_value = line_value_after_label(start_lines, content_line)
    if inline_value:
        content_parts.append(inline_value)
    content_parts.append(collect_content_page_lines(start_lines, float(content_line["bbox"][3]), publication_date_line))
    if publication_date_line is not None:
        return "\n".join(part for part in content_parts if part), first_publication_date(start_lines, float(content_line["bbox"][3]))

    for page_index in range(start_page_index + 1, end_page_index + 1):
        lines = page_lines(doc[page_index])
        publication_date_line = find_publication_date_line(lines)
        content_parts.append(collect_content_page_lines(lines, 60.0, publication_date_line))
        if publication_date_line is not None:
            date = first_publication_date(lines, 0.0)
            break

    if not date:
        date = first_publication_date(start_lines, float(content_line["bbox"][3]))
    return "\n".join(part for part in content_parts if part), date


def label_value_from_lines(lines: list[str], label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.*)$", re.IGNORECASE)
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        value = normalize_cell(match.group(1))
        if value:
            return value
        if index + 1 < len(lines):
            return normalize_cell(lines[index + 1])
    return ""


def month_date_to_iso(value: str) -> str:
    match = PUBLICATION_DATE_RE.search(value)
    if not match:
        return ""
    text = re.sub(
        r"\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$",
        "",
        match.group(0),
        flags=re.IGNORECASE,
    )
    for fmt in ("%B %d, %Y", "%B %d,%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def subject_tags_from_text(text: str) -> list[str]:
    return [
        normalize_cell(match.group(0))
        for match in re.finditer(r"[A-Z][A-Z0-9 &/'.,+-]*\s+\(\d+%\)", text)
    ][:2]


def article_text_from_lines(lines: list[str]) -> str:
    start_index = None
    for index, line in enumerate(lines):
        if re.match(r"^Byline:\s*", line, flags=re.IGNORECASE):
            start_index = index + 1
            break
    if start_index is None:
        for index, line in enumerate(lines):
            if re.match(r"^Publication-Type:\s*", line, flags=re.IGNORECASE):
                start_index = index + 1
                break
    if start_index is None:
        return ""

    stop_index = len(lines)
    for index in range(start_index, len(lines)):
        if re.match(r"^(LANGUAGE|SUBJECT|STATE):\s*", lines[index], flags=re.IGNORECASE):
            stop_index = index
            break
        if lines[index].lower().startswith("copyright "):
            stop_index = index
            break
    return normalize_cell(" ".join(lines[start_index:stop_index]))


def publication_from_content(content: str, description: str, fallback_date: str) -> dict[str, Any]:
    lines = [normalize_cell(line) for line in content.splitlines() if normalize_cell(line)]
    flat = normalize_cell(" ".join(lines))
    load_date = label_value_from_lines(lines, "LOAD-DATE")
    publication_date = ""
    publication_name = ""
    for index, line in enumerate(lines):
        if "LOAD-DATE" in line.upper():
            continue
        match = PUBLICATION_DATE_RE.search(line)
        if not match:
            continue
        publication_date = normalize_cell(match.group(0))
        if index > 0:
            publication_name = lines[index - 1]
        break

    section = label_value_from_lines(lines, "Section")
    length_value = label_value_from_lines(lines, "Length")
    length_match = re.search(r"\d+", length_value)
    publication_type = label_value_from_lines(lines, "Publication-Type") or description
    byline = label_value_from_lines(lines, "Byline")
    language = label_value_from_lines(lines, "LANGUAGE")
    state = label_value_from_lines(lines, "STATE")
    copyright_match = re.search(r"\bCopyright\s+\d{4}\s+.*?(?:All Rights Reserved|$)", flat, flags=re.IGNORECASE)

    return {
        "load_date": load_date,
        "name": publication_name,
        "type": publication_type,
        "date": month_date_to_iso(publication_date or fallback_date),
        "publication_date": publication_date or fallback_date,
        "section": section,
        "length_words": int(length_match.group(0)) if length_match else None,
        "byline": byline,
        "language": language,
        "subject_tags": subject_tags_from_text(label_value_from_lines(lines, "SUBJECT") or flat),
        "state": state,
        "copyright": normalize_cell(copyright_match.group(0)) if copyright_match else "",
        "full_article_text": article_text_from_lines(lines),
    }


def is_global_new_products_title_line(line: dict[str, Any]) -> bool:
    return bool(
        line["is_bold"]
        and float(line["bbox"][1]) < 130
        and float(line["max_size"]) >= 12
        and GLOBAL_NEW_PRODUCTS_TITLE in normalize_spaced_heading(line["text"])
    )


def is_global_new_products_detail(lines: list[dict[str, Any]]) -> bool:
    if any(is_global_new_products_title_line(line) for line in lines):
        return True
    labels = {
        normalize_cell(line["text"]).lower()
        for line in lines
        if line["is_bold"] and float(line["bbox"][0]) < 90 and normalize_cell(line["text"]).endswith(":")
    }
    return "brand:" in labels and "product description:" in labels and "source:" in labels


def is_new_product_label(line: dict[str, Any]) -> bool:
    text = normalize_cell(line["text"])
    return bool(line["is_bold"] and float(line["bbox"][0]) < 110 and text.endswith(":") and len(text) <= 40)


def normalize_new_product_key(label: str) -> str:
    return normalize_cell(label).rstrip(":")


def extract_global_new_products_fields(lines: list[dict[str, Any]], cml_line: dict[str, Any]) -> dict[str, str]:
    label_lines = [
        line
        for line in lines
        if is_new_product_label(line)
        and float(line["bbox"][1]) > float(cml_line["bbox"][3])
        and float(line["bbox"][1]) < 730
    ]
    fields: dict[str, str] = {}
    for index, label_line in enumerate(label_lines):
        key = normalize_new_product_key(label_line["text"])
        next_label = label_lines[index + 1] if index + 1 < len(label_lines) else None
        start_y = float(label_line["bbox"][1])
        end_y = float(next_label["bbox"][1]) if next_label is not None else 730.0
        value_lines = [
            line
            for line in lines
            if start_y - 2 <= float(line["bbox"][1]) < end_y
            and line is not label_line
            and float(line["bbox"][0]) > float(label_line["bbox"][2]) + 8
            and not is_new_product_label(line)
        ]
        fields[key] = normalize_cell(" ".join(line["text"] for line in value_lines))
    return fields


def extract_global_new_products_details(lines: list[dict[str, Any]], cml_line: dict[str, Any]) -> dict[str, Any]:
    fields = extract_global_new_products_fields(lines, cml_line)
    return {
        "description": "Global New Products",
        "Cite": "",
        "date": fields.get("Published", ""),
        "publication": {},
        "global_new_products": fields,
    }


def extract_result_details(
    doc: fitz.Document,
    page_index: int,
    next_record_page_index: int | None,
    last_allowed_page_index: int,
    context: Any = None,
) -> dict[str, str]:
    page = doc[page_index]
    lines = page_lines(page)
    page_width = float(page.rect.width)
    header_lines = [line for line in lines if float(line["bbox"][1]) < 150]
    cml_line = next(
        (
            line
            for line in header_lines
            if CML_RE.fullmatch(line["text"])
            and float(line["bbox"][0]) > page_width * 0.65
        ),
        None,
    )
    if cml_line is None:
        return {}

    cml_match = CML_RE.fullmatch(cml_line["text"])
    mark_candidates = [
        line
        for line in header_lines
        if line is not cml_line
        and line["is_bold"]
        and line_same_row(line, cml_line)
        and float(line["bbox"][0]) < page_width * 0.5
    ]
    mark_text = normalize_cell(mark_candidates[0]["text"]) if mark_candidates else ""

    if is_global_new_products_detail(lines):
        details = extract_global_new_products_details(lines, cml_line)
        return {
            "cml": cml_match.group(1) if cml_match else "",
            "mark_text": mark_text,
            **details,
        }

    description_line = find_label_line(lines, ("Description:",), min_y=float(cml_line["bbox"][3]))
    description = line_value_after_label(lines, description_line) if description_line else ""
    cite_line = find_label_line(
        lines,
        ("Cite:",),
        min_y=float(description_line["bbox"][3]) if description_line else float(cml_line["bbox"][3]),
    )
    content_line = find_label_line(
        lines,
        ("Original Content:", "Content:"),
        min_y=float(cite_line["bbox"][3]) if cite_line else float(description_line["bbox"][3]) if description_line else 0.0,
    )
    cite = normalize_cell(" ".join(collect_lines_between(lines, cite_line, content_line, left_min=120.0))) if cite_line else ""
    content, date = extract_full_content(
        doc,
        page_index,
        content_line,
        next_record_page_index,
        last_allowed_page_index,
    )
    if not date:
        date_min_y = float(content_line["bbox"][3]) if content_line else float(cite_line["bbox"][3]) if cite_line else 0.0
        date = first_publication_date(lines, date_min_y)
    publication = publication_from_content(content, description, date)

    return {
        "cml": cml_match.group(1) if cml_match else "",
        "mark_text": mark_text,
        "description": description,
        "Cite": cite,
        "date": date,
        "publication": publication,
    }


def enrich_records_with_result_details(
    records: list[dict[str, str]],
    doc: fitz.Document,
    page_index_by_footer: dict[int, int],
    search_results_start_index: int | None,
    last_allowed_page_index: int,
    context: Any = None,
) -> list[dict[str, str]]:
    enriched_records: list[dict[str, str]] = []
    for record_index, record in enumerate(records):
        enriched_record = dict(record)
        try:
            logical_page_number = int(record["Page"])
        except ValueError:
            enriched_records.append(enriched_record)
            continue

        page_index = page_index_by_footer.get(logical_page_number)
        next_record_page_index = None
        for next_record in records[record_index + 1 :]:
            try:
                next_record_page_index = page_index_by_footer.get(int(next_record["Page"]))
            except ValueError:
                next_record_page_index = None
            if next_record_page_index is not None:
                break
        if (
            page_index is not None
            and search_results_start_index is not None
            and page_index >= search_results_start_index
        ):
            enriched_record.update(
                extract_result_details(
                    doc,
                    page_index,
                    next_record_page_index,
                    last_allowed_page_index,
                    context=context,
                )
            )
        enriched_records.append(enriched_record)
    return enriched_records


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        module_root_env = Path(__file__).resolve().parents[1] / ".env"
        if module_root_env.exists():
            env_path = module_root_env
        else:
            return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def commercial_text(record: dict[str, str]) -> str:
    publication = record.get("publication", {})
    publication_text = ""
    if isinstance(publication, dict):
        publication_text = " ".join(
            str(publication.get(key, ""))
            for key in ("name", "type", "section", "byline", "language", "state", "full_article_text")
        )
    global_new_products = record.get("global_new_products", {})
    new_product_text = ""
    if isinstance(global_new_products, dict):
        new_product_text = " ".join(str(value) for value in global_new_products.values())
    return normalize_cell(
        " ".join(
            [
                publication_text,
                new_product_text,
                record.get("Cite", ""),
            ]
        )
    )


def sentences_for_text(value: str) -> list[str]:
    text = normalize_cell(value)
    if not text:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def sentence_contains_mark(sentence: str, mark_text: str) -> bool:
    mark = normalize_cell(mark_text)
    if not mark:
        return False
    compact_sentence = re.sub(r"[^a-z0-9]+", "", sentence.lower())
    compact_mark = re.sub(r"[^a-z0-9]+", "", mark.lower())
    return bool(compact_mark and compact_mark in compact_sentence)


def commercial_nouns_near_mark(text: str, mark_text: str) -> list[str]:
    candidates: list[str] = []
    searchable_sentences = [
        sentence for sentence in sentences_for_text(text) if sentence_contains_mark(sentence, mark_text)
    ]
    if not searchable_sentences:
        searchable_sentences = sentences_for_text(text)[:6]
    joined = " ".join(searchable_sentences).lower()
    for noun in COMMERCIAL_NOUNS:
        if re.search(rf"\b{re.escape(noun)}s?\b", joined):
            candidates.append(noun)
    return dedupe_strings(candidates)


def dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_cell(value).strip(" ,.;:-")
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def description_contains(record: dict[str, str], terms: tuple[str, ...]) -> bool:
    description = normalize_cell(record.get("description", "")).lower()
    return any(term in description for term in terms)


def first_label_value(text: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        match = re.search(
            rf"\b{re.escape(label)}:\s*(.+?)(?=\s+[A-Z][A-Za-z /&'-]{{2,40}}:|$)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return normalize_cell(match.group(1))
    return ""


def localized_mark_context(text: str, mark_text: str, radius: int = 1) -> str:
    sentences = sentences_for_text(text)
    if not sentences:
        context = normalize_cell(text[:1200])
    elif any(sentence_contains_mark(s, mark_text) for s in sentences):
        for index, sentence in enumerate(sentences):
            if sentence_contains_mark(sentence, mark_text):
                start = max(0, index - radius)
                end = min(len(sentences), index + radius + 1)
                context = normalize_cell(" ".join(sentences[start:end]))
                break
    else:
        context = normalize_cell(" ".join(sentences[:3]))

    # Apply 70-word limit (Requirement Change #1)
    words = context.split()
    if len(words) > 70:
        context = " ".join(words[:70])
    return context


def deterministic_transcript_owner(text: str) -> str:
    first_lines = normalize_cell(text)
    match = TRANSCRIPT_OWNER_RE.search(first_lines)
    return trim_owner_candidate(match.group(1)) if match else ""


def deterministic_newswire_owner(text: str) -> str:
    match = NEWSWIRE_SOURCE_RE.search(text)
    return trim_owner_candidate(match.group(1)) if match else ""


def geographic_indicators(text: str) -> list[str]:
    values: list[str] = []
    for match in GEOGRAPHIC_INDICATOR_RE.finditer(text):
        values.append(match.group(1))
    return dedupe_strings(values)


def deterministic_layer1_triage(record: dict[str, str]) -> dict[str, Any]:
    text = commercial_text(record)
    if description_contains(record, TRACK_A_DESCRIPTION_TERMS):
        owner_name = first_label_value(
            text,
            ("Company Name", "Owner", "Registrant", "Applicant", "Business Name"),
        )
        goods_services = first_label_value(
            text,
            ("Line of Business", "Business Description", "SIC Code", "Goods/Services", "Goods and Services"),
        )
        return {
            "layer1_track": "A",
            "layer1_behavior": "terminal_rule_based",
            "layer2_route": "bypass",
            "deterministic_owner_name": owner_name,
            "deterministic_goods_services": goods_services,
        }

    if description_contains(record, TRACK_B_DESCRIPTION_TERMS):
        owner_name = deterministic_transcript_owner(text)
        if not owner_name:
            owner_name = deterministic_newswire_owner(text)
        return {
            "layer1_track": "B",
            "layer1_behavior": "partial_extraction",
            "layer2_route": "goods_only",
            "deterministic_owner_name": owner_name,
            "deterministic_goods_services": "",
            "localized_mark_context": localized_mark_context(text, record.get("mark_text", "")),
        }

    if description_contains(record, TRACK_C_DESCRIPTION_TERMS):
        return {
            "layer1_track": "C",
            "layer1_behavior": "core_metadata",
            "layer2_route": "full",
            "deterministic_owner_name": "",
            "deterministic_goods_services": "",
            "geographic_indicators": geographic_indicators(text),
        }

    return {
        "layer1_track": "D",
        "layer1_behavior": "safe_fail",
        "layer2_route": "fallback",
        "deterministic_owner_name": "",
        "deterministic_goods_services": "",
    }


def is_rejected_owner(value: str) -> bool:
    text = normalize_cell(value)
    lower = text.lower()
    if not text:
        return True
    if lower in OWNER_STOPWORDS:
        return True
    if re.match(r"^(?:i|you|we|they|he|she|it)\s+\w+\b", lower):
        return True
    if re.fullmatch(r"(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/.*)?", lower):
        return True
    return any(term in lower for term in PUBLICATION_OWNER_TERMS)


def trim_owner_candidate(value: str) -> str:
    candidate = normalize_cell(value)
    candidate = re.split(
        r"\b(?:LOAD-DATE|Section|Length|Publication-Type|Byline|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        candidate,
        maxsplit=1,
    )[0]
    candidate = re.sub(r"^(?:The|A|An)\s+", "", candidate).strip(" ,.;:-")
    return candidate


def extract_owner_candidates(record: dict[str, str]) -> list[str]:
    text = commercial_text(record)
    mark_text = record.get("mark_text", "")
    candidates: list[str] = []

    for pattern in OWNER_PATTERN_RE_LIST:
        for match in pattern.finditer(text):
            groups = [group for group in match.groups() if group]
            candidates.extend(trim_owner_candidate(group) for group in groups)

    if mark_text:
        escaped_mark = re.escape(mark_text)
        for pattern in (
            re.compile(rf"\b{escaped_mark}\b\s+from\s+([A-Z][A-Za-z0-9&'. -]{{2,90}}?)\b", re.IGNORECASE),
            re.compile(rf"\b{escaped_mark}\b\s+by\s+([A-Z][A-Za-z0-9&'. -]{{2,90}}?)\b", re.IGNORECASE),
        ):
            for match in pattern.finditer(text):
                candidates.append(trim_owner_candidate(match.group(1)))

    filtered = [candidate for candidate in candidates if not is_rejected_owner(candidate)]
    return dedupe_strings(filtered)[:8]


def extract_goods_candidates(record: dict[str, str]) -> list[str]:
    text = commercial_text(record)
    mark_text = record.get("mark_text", "")
    candidates = commercial_nouns_near_mark(text, mark_text)
    lower_text = text.lower()
    category_map = {
        "ice cream": "ice cream",
        "hard seltzer": "alcoholic beverages",
        "beer": "alcoholic beverages",
        "wine": "alcoholic beverages",
        "cocktail": "cocktails",
        "restaurant": "restaurant services",
        "smoothie": "smoothies",
        "smoothies": "smoothies",
        "cannabis": "cannabis products",
        "vape": "cannabis products",
        "flower jars": "cannabis products",
        "tea": "tea",
        "coffee": "coffee",
        "dessert": "desserts",
        "snack": "snacks",
        "food": "food",
        "drink": "beverages",
        "beverage": "beverages",
    }
    candidates.extend(category for signal, category in category_map.items() if signal in lower_text)
    return dedupe_strings(candidates)[:10]


def deterministic_candidate_analysis(
    owner_candidates: list[str],
    goods_candidates: list[str],
) -> dict[str, Any]:
    owner_name = next((owner for owner in owner_candidates if not is_rejected_owner(owner)), None)
    goods_services = ", ".join(goods_candidates[:3]) if goods_candidates else None
    return {
        "owner_name": owner_name,
        "goods_services": goods_services,
    }


_AZURE_CLIENT_LOCK = threading.Lock()
_CACHED_AZURE_CLIENT = None
_CACHED_INSTRUCTOR_CLIENT = None
_CACHED_MODEL = None


def openai_client_and_model() -> tuple[Any, str] | tuple[None, str]:
    global _CACHED_AZURE_CLIENT, _CACHED_MODEL

    if _CACHED_AZURE_CLIENT is not None:
        return _CACHED_AZURE_CLIENT, _CACHED_MODEL

    with _AZURE_CLIENT_LOCK:
        if _CACHED_AZURE_CLIENT is not None:
            return _CACHED_AZURE_CLIENT, _CACHED_MODEL

        load_env_file()
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip('"')
        model = (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or ""
        ).strip('"')
        if not api_key or not endpoint or not api_version:
            return None, model
        if not model:
            raise RuntimeError(
                "Missing Azure OpenAI model configuration. "
                "Set AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_MODEL."
            )
        responses_base_url = endpoint if endpoint.endswith("/openai/v1") else f"{endpoint}/openai/v1"
        client = AzureOpenAI(
            api_key=api_key,
            base_url=responses_base_url,
            api_version=api_version,
            timeout=_LLM_TIMEOUT_SECONDS,
        )
        client._custom_query = {}

        _CACHED_AZURE_CLIENT = client
        _CACHED_MODEL = model
        return _CACHED_AZURE_CLIENT, _CACHED_MODEL


def instructor_client_and_model() -> tuple[Any, str] | tuple[None, str]:
    global _CACHED_INSTRUCTOR_CLIENT, _CACHED_MODEL

    if _CACHED_INSTRUCTOR_CLIENT is not None:
        return _CACHED_INSTRUCTOR_CLIENT, _CACHED_MODEL

    # Initialize the base Azure client first (this has its own internal lock)
    client, model = openai_client_and_model()
    if client is None:
        return None, model

    with _AZURE_CLIENT_LOCK:
        if _CACHED_INSTRUCTOR_CLIENT is not None:
            return _CACHED_INSTRUCTOR_CLIENT, _CACHED_MODEL
        _CACHED_INSTRUCTOR_CLIENT = instructor.from_openai(client, mode=instructor.Mode.RESPONSES_TOOLS)
        _CACHED_MODEL = model
        return _CACHED_INSTRUCTOR_CLIENT, _CACHED_MODEL


def compact_record_block(record: dict[str, Any], limit: int = 7000) -> str:
    publication = record.get("publication", {})
    global_new_products = record.get("global_new_products", {})
    article_text = ""
    if isinstance(publication, dict):
        article_text = normalize_cell(str(publication.get("full_article_text", "")))
    
    # Word-based truncation (Requirement 4, 5, 6)
    words = article_text.split()
    if len(words) > 70:
        article_text = " ".join(words[:70])

    if len(article_text) > limit:
        article_text = article_text[:limit]

    # Create LLM-safe publication copy to avoid duplicate article text (Required Change #2)
    publication_for_llm = dict(publication)
    if isinstance(publication_for_llm, dict):
        publication_for_llm.pop("full_article_text", None)

    return f"""MARK TEXT:
{record.get("mark_text", "")}

DESCRIPTION:
{record.get("description", "")}

CITE:
{record.get("Cite", "")}

PUBLICATION:
{json.dumps(publication_for_llm, ensure_ascii=False)}

GLOBAL NEW PRODUCTS:
{json.dumps(global_new_products, ensure_ascii=False)}

FULL ARTICLE TEXT:
{article_text}"""


def layer2_prompt(
    record: dict[str, Any],
    triage: dict[str, Any],
    owner_candidates: list[str],
    goods_candidates: list[str],
) -> str:
    track = triage.get("layer1_track", "D")
    deterministic_owner = triage.get("deterministic_owner_name", "")
    
    input_payload = {
        "routing_track": track,
        "target_mark": record.get("mark_text", ""),
        "document_description": record.get("description", ""),
        "layer1_deterministic_owner": deterministic_owner if deterministic_owner else "None Identified",
        "heuristic_owner_candidates": owner_candidates,
        "heuristic_goods_candidates": goods_candidates,
    }
    

    if track == "B":
        input_payload["evidence_text"] = triage.get("localized_mark_context", "")
        track_instructions = (
            "This is a Track B corporate/press document. Layer 1 has already executed structural analysis.\n"
            "If 'layer1_deterministic_owner' is provided, cross-verify it against the text and preserve it as the 'owner_name'.\n"
            "Focus your evaluation on extracting the specific commercial items or services tied directly to the mark."
        )
    elif track == "C":
        input_payload["evidence_text"] = compact_record_block(record)
        input_payload["geographic_metadata"] = triage.get(
            "geographic_indicators",
            [],
        )

        track_instructions = """TRACK C

Source type: newspaper, magazine, web publication, or media article.

Commercial ownership must be separated from editorial ownership.

Ignore editorial entities unless they are explicitly operating the branded goods or services.

Examples of entities that are normally not owners:
- journalists
- article authors
- publishers
- newspapers
- magazines
- news agencies
- web portals
Search for the entity actually using the mark in commerce.

When multiple plausible commercial entities exist, select the entity with the strongest direct relationship to the branded goods or services.

Evaluate ownership evidence in this order:
1. manufacturer
2. brand owner
3. distributor
4. service provider
5. retailer
Use geographic metadata only as supporting evidence.
If ownership is not stated directly, select the most commercially plausible owner supported by the article narrative and commercial context.
If goods or services are not stated directly, derive the most commercially reasonable commercial description from the article narrative and commercial signals.
Do not invent facts that are absent from the supplied evidence.""".strip()
    else:
        input_payload["evidence_text"] = compact_record_block(record)

        track_instructions = """TRACK D
Source type: unclassified evidence.
Use only commercially relevant evidence.
Strong ownership evidence includes:
- manufactured by
- distributed by
- launched by
- sold by
- operated by
- offered by
- produced by
- marketed by

When multiple plausible commercial entities exist, select the entity with the strongest direct relationship to the branded goods or services.

Do not infer ownership solely from mention frequency, article prominence, or proximity to the mark.
Select the entity with the strongest commercial relationship to the mark.
If ownership is ambiguous, choose the most commercially plausible entity supported by the available commercial evidence.
If goods or services are ambiguous, derive the most commercially reasonable commercial description supported by the available evidence.
Do not invent entities, products, services, or facts that do not appear in the supplied evidence.""".strip()

    prompt = f"""{track_instructions}

DATA STRUCTURE FOR ANALYSIS:
{json.dumps(input_payload, indent=2, ensure_ascii=False)}

OUTPUT REQUIREMENT:

Return only:

{{
"owner_name": string | null,
"goods_services": string | null,
"nice_class": integer | null
}}

Do not return markdown.

Do not return explanatory text."""
    return prompt


def call_llm_analysis(
    record: dict[str, Any],
    triage: dict[str, Any],
    owner_candidates: list[str],
    goods_candidates: list[str],
) -> tuple[dict[str, Any] | None, str]:
    if triage.get("layer1_track") == "A":
        return None, "bypassed"
    iclient, model = instructor_client_and_model()
    if iclient is None:
        return None, "not_configured"
    try:
        response = iclient.responses.create(
            model=model,
            response_model=CorsearchAnalysis,
            input=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": layer2_prompt(record, triage, owner_candidates, goods_candidates),
                },
            ],
        )
        parsed = response.model_dump()
        return (parsed, "used")
    except (ValidationError, InstructorRetryException, IncompleteOutputException, ResponseParsingError):
        # All structured-output validation or parsing failures
        return None, "invalid_json"
    except Exception:
        # API failure, Timeout, Authentication failure, Network failure -> return None, "failed"
        return None, "failed"


def valid_goods_services(value: str, mark_text: str) -> bool:
    text = normalize_cell(value)
    if not text:
        return False
    if text.lower() in GENERIC_GOODS_VALUES:
        return False
    compact_text = re.sub(r"[^a-z0-9]+", "", text.lower())
    compact_mark = re.sub(r"[^a-z0-9]+", "", mark_text.lower())
    return not bool(compact_text and compact_text == compact_mark)


def post_validate_analysis(analysis: dict[str, Any], mark_text: str) -> dict[str, Any]:
    owner_name = analysis.get("owner_name")
    goods_services = analysis.get("goods_services")
    nice_class = analysis.get("nice_class")
    
    if owner_name is not None:
        owner_name = normalize_cell(owner_name)
        if is_rejected_owner(owner_name):
            owner_name = None
    if goods_services is not None:
        goods_services = normalize_cell(goods_services)
        if not valid_goods_services(goods_services, mark_text):
            goods_services = None
            
    # Validate the Nice Class constraint (Valid integer from 1 to 45)
    if nice_class is not None:
        try:
            nice_class_int = int(nice_class)
            if 1 <= nice_class_int <= 45:
                nice_class = nice_class_int
            else:
                nice_class = None
        except (ValueError, TypeError):
            nice_class = None

    return {
        "owner_name": owner_name,
        "goods_services": goods_services,
        "nice_class": nice_class,
    }


def analyze_commercial_record(record: dict[str, str]) -> dict[str, Any]:
    enriched = dict(record)

    triage = deterministic_layer1_triage(enriched)
    owner_candidates = extract_owner_candidates(enriched)
    goods_candidates = extract_goods_candidates(enriched)
    deterministic_owner = normalize_cell(triage.get("deterministic_owner_name", ""))
    deterministic_goods = normalize_cell(triage.get("deterministic_goods_services", ""))
    if deterministic_owner and not is_rejected_owner(deterministic_owner):
        owner_candidates = dedupe_strings([deterministic_owner, *owner_candidates])
    if deterministic_goods:
        goods_candidates = dedupe_strings([deterministic_goods, *goods_candidates])
    fallback = deterministic_candidate_analysis(owner_candidates, goods_candidates)
    llm_analysis, layer2_status = call_llm_analysis(enriched, triage, owner_candidates, goods_candidates)
    
    # Run the expanded post-validation logic
    final_analysis = post_validate_analysis(llm_analysis or fallback, enriched.get("mark_text", ""))
    final_owner = final_analysis["owner_name"]
    final_goods = final_analysis["goods_services"]
    final_nice_class = final_analysis.get("nice_class")
    
    if not final_owner and deterministic_owner and triage["layer1_track"] in {"A", "B"}:
        final_owner = deterministic_owner
    if not final_goods and deterministic_goods and triage["layer1_track"] == "A":
        final_goods = deterministic_goods

    enriched.update(
        {
            "owner_name": final_owner,
            "goods_services": final_goods,
            "nice_class": final_nice_class, # Injects validated nice class directly into the parsed records mapping
            "owner_candidates": owner_candidates,
            "goods_candidates": goods_candidates,
            "layer2_status": layer2_status,
            **triage,
        }
    )
    for retired_field in (
        "cite",
        "cml_unit_id",
        "page_number",
        "owner_confidence",
        "goods_confidence",
        "commercial_relevance_score",
        "commercial_usage_type",
        "reasoning",
        "content",
        "original_content",
    ):
        enriched.pop(retired_field, None)
    return enriched

#decrease
def analyze_commercial_records(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=5) as executor:
        final_results = list(executor.map(analyze_commercial_record, records))
    return final_results


def extract_pdf(pdf_path: str, context: Any = None) -> dict[str, Any]:
    Web_corsearch_start_page, Web_corsearch_end_page = calculate_page_range(pdf_path, context=context)
    records: list[dict[str, str]] = []

    try:
        start_page = int(Web_corsearch_start_page)
        end_page = int(Web_corsearch_end_page)
    except ValueError:
        return {
            "return_type": "common law",
            "vendor_name": "Corsearch",
            "Web_corsearch_start_page": Web_corsearch_start_page,
            "Web_corsearch_end_page": Web_corsearch_end_page,
            "records": records,
        }

    last_allowed_page = end_page - 1

    doc = context.get_fitz_doc() if context else fitz.open(pdf_path)
    pdf = context.get_pdfplumber_doc() if context else pdfplumber.open(pdf_path)
    try:
        start_page_index = locate_start_page_index(doc, start_page)
        if start_page_index is not None:
            page_index_by_footer = build_page_index_by_footer(
                doc,
                start_page_index,
                start_page,
                last_allowed_page,
            )
            search_results_start_index = locate_search_results_start_index(
                doc,
                start_page_index,
                last_allowed_page,
            )
            last_allowed_page_index = page_index_by_footer.get(last_allowed_page, doc.page_count - 1)
            for page_index in range(start_page_index, doc.page_count):
                logical_page_number = footer_page_number(doc[page_index])
                if logical_page_number is None:
                    continue
                if logical_page_number > last_allowed_page:
                    break
                if logical_page_number < start_page:
                    continue
                if not is_summary_page(doc[page_index]):
                    continue
                if page_index >= len(pdf.pages):
                    break

                page_records = extract_table_records(pdf.pages[page_index])
                page_records.extend(extract_position_records(pdf.pages[page_index]))
                records.extend(page_records)
            records = enrich_records_with_result_details(
                dedupe_records(records),
                doc,
                page_index_by_footer,
                search_results_start_index,
                last_allowed_page_index,
                context=context,
            )
            records = analyze_commercial_records(records)
    finally:
        if not context:
            doc.close()
            pdf.close()

    return {
        "return_type": "common law",
        "vendor_name": "Corsearch",
        "Web_corsearch_start_page": Web_corsearch_start_page,
        "Web_corsearch_end_page": Web_corsearch_end_page,
        "records": records,
    }


def timestamped_output_path(pdf_path: str) -> Path:
    output_dir = Path("corsearch_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{Path(pdf_path).stem.replace(' ', '_')}_cml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def save_json_output(pdf_path: str, output_path: str | None = None) -> Path:
    data = extract_pdf(pdf_path)
    path = Path(output_path) if output_path else timestamped_output_path(pdf_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Corsearch Common Law CML results and save JSON.")
    parser.add_argument("pdf_path", help="Input Corsearch PDF path.")
    parser.add_argument("-o", "--output", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(save_json_output(args.pdf_path, args.output))


if __name__ == "__main__":
    main()
