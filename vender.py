# "vender_detection_state.py" - location:  vendor_pipeline\
import argparse
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

# Add src to sys.path to resolve 'src' based imports correctly
# Current file is in src/extractors/vendor_pipeline/vender_detection_state.py
# We want to add 'src' directory, which is 3 levels up from this file's directory.
_current_dir = Path(__file__).resolve().parent
_src_root = _current_dir.parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from datetime import datetime
from pathlib import Path

from statistics import median
from typing import Any, Dict
from dataclasses import dataclass, field

import orjson
import threading
import io
import concurrent.futures
import fitz

_INITIALIZATION_LOCK = threading.Lock()
_MODULES_INITIALIZED = False

@dataclass
class DocumentContext:
    pdf_path: str
    pdf_bytes: bytes = field(repr=False)

    vendor_type: str
    clarivate_ranges: Dict[str, str] = field(default_factory=dict)
    page_cache: Dict[int, Any] = field(default_factory=dict)
    _local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    def get_fitz_doc(self) -> fitz.Document:
        if not hasattr(self._local, "fitz_doc") or self._local.fitz_doc.is_closed:
            self._local.fitz_doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        return self._local.fitz_doc

    def get_pdfplumber_doc(self) -> Any:
        if not hasattr(self._local, "pdfplumber_doc"):
            import pdfplumber
            self._local.pdfplumber_doc = pdfplumber.open(io.BytesIO(self.pdf_bytes))
        return self._local.pdfplumber_doc

    def close(self) -> None:
        if hasattr(self._local, "fitz_doc") and not self._local.fitz_doc.is_closed:
            self._local.fitz_doc.close()
        if hasattr(self._local, "pdfplumber_doc"):
            self._local.pdfplumber_doc.close()




REQUIRED_ENV_NAME = "extraction_state_low"
DEFAULT_PDF_PATH = "corsearch/Search Report - VOLCANIX.PDF"
OUTPUT_KEYS = {
    "vendor_name": "error",
    "mark_searched": "error",
    "classes_searched": "error",
    "goods_services_searched": "error",
    "state_starting_page": "error",
    "state_end_page": "error",
}
CLARIVATE_OUTPUT_KEYS = {
    "vender_name": "error",
    "Mark Searched": "error",
    "Classes Searched": "error",
    "Goods/Services Searched": "error",
    "state_starting_page": "error",
    "state_end_page": "error",
}
CORSEARCH_OUTPUT_KEYS = {
    "vender_name": "error",
    "Mark": "error",
    "Classes Searched": "error",
    "Goods/Services Searched": "error",
    "state_starting_page": "error",
    "state_end_page": "error",
}
VENDOR_COMPUMARK = "CompuMark"
VENDOR_CLARIVATE = "Clarivate"
VENDOR_CORSEARCH = "Corsearch"
VENDOR_FOVEA = "Fovea"
RETURN_TYPE_STATE_LAW = "state law"
RETURN_TYPE_COMMON_LAW = "common law"
RETURN_TYPE_BUSINESS_DOMAIN = "B_&_D"
RETURN_TYPE_CML_CORSEARCH = "common_law_corsearch"
WEB_COMMON_LAW_SECTION = "web_common_law"
COMMON_LAW_DATABASE_SECTION = "common_law_database"
REPORT_TITLE_RE = re.compile(r"^(Trademark Research Report|Title Research Report|Research Report)$")
FOOTER_PAGE_RE = re.compile(r"^Page:\s*(\d+)$")
CORSEARCH_FOOTER_PAGE_RE = re.compile(r"^Page\s+(\d+)$", re.IGNORECASE)
BARE_FOOTER_PAGE_RE = re.compile(r"^\d+$")
TOC_TITLE = "Table of Contents"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def validate_input_file(file_path: str) -> bool:
    """Centrally validates file existence, extension, and integrity before processing."""
    # Case 1: File Does Not Exist
    if not os.path.exists(file_path):
        print(f"[VALIDATION] File not found: {file_path}")
        print("[VALIDATION] Extraction aborted.")
        return False

    # Case 2: Unsupported Extension
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in {".pdf", ".docx"}:
        print(f"[VALIDATION] Unsupported file type: {file_ext}")
        print("[VALIDATION] Supported formats: PDF, DOCX")
        print("[VALIDATION] Extraction aborted.")
        return False

    # Case 3: Invalid PDF integrity check
    if file_ext == ".pdf":
        try:
            with fitz.open(file_path) as doc:
                if doc.page_count == 0:
                    print("[VALIDATION] Invalid PDF file: Document has no pages.")
                    print("[VALIDATION] Extraction aborted.")
                    return False
        except Exception:
            print("[VALIDATION] Invalid PDF file: Document is corrupted or unreadable.")
            print("[VALIDATION] Extraction aborted.")
            return False

    return True


def validate_return_type(return_type: str) -> None:
    pass


def require_conda_env() -> None:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name and env_name != REQUIRED_ENV_NAME:
        raise RuntimeError(
            f"This extractor must run in conda env '{REQUIRED_ENV_NAME}', "
            f"but CONDA_DEFAULT_ENV is '{env_name}'."
        )


def is_bold_font(font_name: str) -> bool:
    return "bold" in font_name.lower() or "black" in font_name.lower()


def page_lines(page: fitz.Page) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not normalize_text(text):
                    continue
                spans.append(
                    {
                        "text": text,
                        "font": span.get("font", ""),
                        "size": float(span.get("size", 0.0)),
                        "color": int(span.get("color", 0)),
                        "bbox": tuple(float(v) for v in span["bbox"]),
                    }
                )
            if not spans:
                continue
            text = normalize_text(" ".join(normalize_text(span["text"]) for span in spans))
            sizes = [span["size"] for span in spans]
            fonts = [span["font"] for span in spans]
            colors = [span["color"] for span in spans]
            lines.append(
                {
                    "text": text,
                    "bbox": tuple(float(v) for v in line["bbox"]),
                    "spans": spans,
                    "max_size": max(sizes),
                    "colors": colors,
                    "is_bold": any(is_bold_font(font) for font in fonts),
                }
            )
    return sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def same_or_next_line_value(lines: list[dict[str, Any]], label: str) -> str:
    for index, line in enumerate(lines):
        text = line["text"]
        if text.startswith(label):
            value = normalize_text(text[len(label) :])
            if value:
                return value
            if index + 1 < len(lines):
                return normalize_text(lines[index + 1]["text"])
    return "error"


def corsearch_multiline_value_after_label(
    lines: list[dict[str, Any]],
    label: str,
    page_width: float,
) -> str:
    for index, line in enumerate(lines):
        if line["text"] != label:
            continue

        values: list[str] = []
        same_row_values = [
            next_line
            for next_line in lines
            if next_line is not line
            and next_line["bbox"][0] > line["bbox"][2]
            and line_same_row(line, next_line)
        ]
        values.extend(next_line["text"] for next_line in sorted(same_row_values, key=lambda item: item["bbox"][0]))
        label_y1 = line["bbox"][3]
        value_left = page_width * 0.18
        previous_bottom: float | None = None
        for next_line in lines[index + 1 :]:
            text = next_line["text"]
            x0, y0, _x1, _y1 = next_line["bbox"]
            if y0 <= label_y1:
                continue
            if x0 < value_left and next_line["is_bold"]:
                break
            if previous_bottom is not None and y0 - previous_bottom > 35:
                break
            values.append(text)
            previous_bottom = next_line["bbox"][3]
        if values:
            return "\n".join(values)

    return "error"


def extract_corsearch_mark(lines: list[dict[str, Any]], page_width: float) -> str:
    right_half_lines = [line for line in lines if line["bbox"][0] >= page_width * 0.5]
    for index, line in enumerate(right_half_lines):
        if line["text"] != "Mark:":
            continue
        if index + 1 < len(right_half_lines):
            return right_half_lines[index + 1]["text"]
    return "error"


def extract_corsearch_classes(goods_services: str) -> str:
    if goods_services == "error":
        return "error"
    match = re.search(r"Nice\s+Class(?:\(es\))?:\s*([0-9,\s]+)", goods_services, flags=re.IGNORECASE)
    if not match:
        return "error"
    classes = normalize_text(match.group(1))
    return classes.rstrip(",") if classes else "error"


def extract_corsearch_cover_page(page: fitz.Page, vendor_name: str) -> dict[str, str]:
    lines = page_lines(page)
    page_width = float(page.rect.width)
    goods_services = corsearch_multiline_value_after_label(lines, "Goods/Services:", page_width)
    return {
        "vender_name": vendor_name,
        "Mark": extract_corsearch_mark(lines, page_width),
        "Classes Searched": extract_corsearch_classes(goods_services),
        "Goods/Services Searched": goods_services,
    }


def normalize_vendor_for_comparison(value: str) -> str:
    text = normalize_text(value)
    text = text.replace("\u2122", "")
    text = re.sub(r"\btm\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    return text.lower()


def is_clarivate_vendor_name(value: str) -> bool:
    return normalize_vendor_for_comparison(value) == "clarivate"


def is_corsearch_vendor_name(value: str) -> bool:
    return normalize_vendor_for_comparison(value) == "corsearch"


def line_is_dark(line: dict[str, Any]) -> bool:
    for color in line.get("colors", []):
        red = (color >> 16) & 255
        green = (color >> 8) & 255
        blue = color & 255
        if max(red, green, blue) <= 80:
            return True
    return False


def is_fovea_vendor(page: fitz.Page) -> bool:
    """Detects if the PDF is a Fovea document by checking the first page."""
    text = page.get_text() or ""
    
    # 1. Search the entire page, converted to lowercase
    footer_text_lower = text.lower()
    
    # 2. Strip all whitespace and newlines to prevent PDF extraction spacing errors (e.g. "f o v e a")
    compressed_text = re.sub(r"\s+", "", footer_text_lower)
    
    # 3. Check against keywords
    keyword_list = ["fovea", "foveaip", "www.foveaip.com", "foveaip.com", "www.foveaip"]
    
    for keyword in keyword_list:
        if keyword in compressed_text:
            return True
            
    return False

def extract_clarivate_vendor_name(page: fitz.Page) -> str:
    lines = page_lines(page)
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    page_text = " ".join(line["text"] for line in lines)
    if re.search(r"\b(Mark Searched:|Classes Searched:|Goods/Services:|Goods and Services:)", page_text):
        return "error"
    title_lines = [
        line
        for line in lines
        if REPORT_TITLE_RE.match(line["text"])
        and line["bbox"][1] <= page_height * 0.45
    ]
    if not title_lines:
        return "error"

    title_top = min(line["bbox"][1] for line in title_lines)
    page_largest_size = max((line["max_size"] for line in lines), default=0.0)
    text_candidates = [
        line
        for line in lines
        if line["bbox"][1] < title_top
        and line["bbox"][1] <= page_height * 0.32
        and line["bbox"][0] < page_width * 0.5
        and line["is_bold"]
        and line["max_size"] >= page_largest_size - 0.5
        and any(char.isalpha() for char in line["text"])
    ]
    for candidate in text_candidates:
        if is_clarivate_vendor_name(candidate["text"]):
            return candidate["text"]

    # In Clarivate cover pages the logo text is embedded as a header image.
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 1:
            continue
        x0, y0, x1, y1 = (float(v) for v in block["bbox"])
        width = x1 - x0
        height = y1 - y0
        if (
            y1 < title_top
            and y0 <= page_height * 0.2
            and x0 < page_width * 0.5
            and width >= page_width * 0.15
            and height <= page_height * 0.08
        ):
            return "Clarivate"

    return "error"


def corsearch_has_top_left_logo_region(page: fitz.Page) -> bool:
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    for block in page.get_text("dict")["blocks"]:
        x0, y0, x1, y1 = (float(v) for v in block["bbox"])
        width = x1 - x0
        height = y1 - y0
        if (
            block.get("type") == 1
            and x0 <= page_width * 0.18
            and y0 <= page_height * 0.12
            and width >= page_width * 0.12
            and height <= page_height * 0.08
        ):
            return True
    return False


def extract_corsearch_vendor_name(page: fitz.Page) -> str:
    lines = page_lines(page)
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    header_candidates = [
        line
        for line in lines
        if line["bbox"][0] <= page_width * 0.35
        and line["bbox"][1] <= page_height * 0.18
        and line["is_bold"]
        and any(char.isalpha() for char in line["text"])
    ]
    for candidate in sorted(header_candidates, key=lambda item: (-item["max_size"], item["bbox"][1], item["bbox"][0])):
        if is_corsearch_vendor_name(candidate["text"]):
            return candidate["text"]

    page_text = " ".join(line["text"] for line in lines)
    if corsearch_has_top_left_logo_region(page) and re.search(r"\bCorsearch\b", page_text, flags=re.IGNORECASE):
        return "Corsearch"

    return "error"


def extract_vendor_name(lines: list[dict[str, Any]], page_width: float, page_height: float) -> str:
    title_lines = [
        line
        for line in lines
        if REPORT_TITLE_RE.match(line["text"])
        and line["bbox"][1] <= page_height * 0.45
    ]
    if not title_lines:
        return "error"

    title_top = min(line["bbox"][1] for line in title_lines)
    page_largest_size = max((line["max_size"] for line in lines), default=0.0)
    header_candidates = [
        line
        for line in lines
        if line["bbox"][1] < title_top
        and line["bbox"][1] <= page_height * 0.25
        and line["bbox"][0] < page_width * 0.5
        and line["is_bold"]
        and line["max_size"] >= page_largest_size - 0.5
        and any(char.isalpha() for char in line["text"])
    ]
    if not header_candidates:
        fallback_size = max(
            (line["max_size"] for line in lines if line["bbox"][1] < title_top),
            default=0.0,
        )
        header_candidates = [
            line
            for line in lines
            if line["bbox"][1] < title_top
            and line["bbox"][1] <= page_height * 0.25
            and line["bbox"][0] < page_width * 0.5
            and line["is_bold"]
            and line["max_size"] >= fallback_size - 0.5
            and any(char.isalpha() for char in line["text"])
        ]
    if not header_candidates:
        page_text = " ".join(line["text"] for line in lines)
        if re.search(r"\bCompuMark\b", page_text):
            return "CompuMark"
        return "error"
    return normalize_text(header_candidates[0]["text"])


def extract_cover_page(page: fitz.Page) -> dict[str, str]:
    lines = page_lines(page)
    return {
        "vendor_name": extract_vendor_name(lines, float(page.rect.width), float(page.rect.height)),
        "mark_searched": same_or_next_line_value(lines, "Mark Searched:"),
        "classes_searched": same_or_next_line_value(lines, "Classes Searched:"),
        "goods_services_searched": same_or_next_line_value(lines, "Goods/Services:"),
    }


def footer_page_number(lines: list[dict[str, Any]], page_width: float, page_height: float) -> int | None:
    for line in lines:
        match = FOOTER_PAGE_RE.match(line["text"])
        if not match:
            continue
        x0, y0, _x1, _y1 = line["bbox"]
        if x0 >= page_width * 0.65 and y0 >= page_height * 0.85:
            return int(match.group(1))
    return None


def clarivate_footer_page_number(lines: list[dict[str, Any]], page_width: float, page_height: float) -> int | None:
    footer_candidates: list[tuple[float, int]] = []
    for line in lines:
        text = line["text"]
        if not BARE_FOOTER_PAGE_RE.match(text):
            continue
        x0, y0, _x1, _y1 = line["bbox"]
        if x0 >= page_width * 0.85 and y0 >= page_height * 0.9:
            footer_candidates.append((x0, int(text)))
    if not footer_candidates:
        return None
    return max(footer_candidates, key=lambda item: item[0])[1]


def corsearch_footer_page_number(lines: list[dict[str, Any]], page_width: float, page_height: float) -> int | None:
    footer_candidates: list[tuple[float, int]] = []
    for line in lines:
        match = CORSEARCH_FOOTER_PAGE_RE.match(line["text"])
        if not match:
            continue
        x0, y0, _x1, _y1 = line["bbox"]
        if x0 >= page_width * 0.75 and y0 >= page_height * 0.9:
            footer_candidates.append((x0, int(match.group(1))))
    if not footer_candidates:
        return None
    return max(footer_candidates, key=lambda item: item[0])[1]


def find_clarivate_page_by_footer(doc: fitz.Document, footer_number: int) -> fitz.Page | None:
    for page in doc:
        lines = page_lines(page)
        page_number = clarivate_footer_page_number(
            lines,
            float(page.rect.width),
            float(page.rect.height),
        )
        if page_number == footer_number:
            return page
    return None


def find_corsearch_page_by_footer(doc: fitz.Document, footer_number: int) -> fitz.Page | None:
    for page in doc:
        lines = page_lines(page)
        page_number = corsearch_footer_page_number(
            lines,
            float(page.rect.width),
            float(page.rect.height),
        )
        if page_number == footer_number:
            return page
    return None


def has_toc_border(page: fitz.Page, title_line: dict[str, Any]) -> bool:
    title_box = fitz.Rect(title_line["bbox"])
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if not rect:
            continue
        if rect.width < page.rect.width * 0.6 or rect.height < page.rect.height * 0.5:
            continue
        if rect.contains(title_box):
            return True
    return False


def is_table_of_contents_page(page: fitz.Page) -> bool:
    lines = page_lines(page)
    sizes = [line["max_size"] for line in lines]
    typical_size = median(sizes) if sizes else 0.0
    page_width = float(page.rect.width)

    for line in lines:
        if line["text"] != TOC_TITLE:
            continue
        x0, _y0, x1, _y1 = line["bbox"]
        center_x = (x0 + x1) / 2
        centered = abs(center_x - page_width / 2) <= page_width * 0.12
        visually_primary = line["is_bold"] and line["max_size"] >= max(11.0, typical_size + 1.0)
        if centered and visually_primary and has_toc_border(page, line):
            return True
        if centered and visually_primary:
            return True
    return False


def is_clarivate_table_of_contents_page(page: fitz.Page) -> bool:
    lines = page_lines(page)
    sizes = [line["max_size"] for line in lines]
    typical_size = median(sizes) if sizes else 0.0
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    for line in lines:
        if line["text"] != TOC_TITLE:
            continue
        x0, y0, _x1, _y1 = line["bbox"]
        in_left_header = x0 < page_width * 0.35 and y0 <= page_height * 0.12
        visually_primary = line["is_bold"] and line["max_size"] >= max(14.0, typical_size + 4.0)
        if in_left_header and visually_primary and line_is_dark(line):
            return True
    return False


def is_corsearch_table_of_contents_page(page: fitz.Page) -> bool:
    lines = page_lines(page)
    sizes = [line["max_size"] for line in lines]
    typical_size = median(sizes) if sizes else 0.0
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    for line in lines:
        compact_text = re.sub(r"[^A-Za-z]+", "", line["text"]).upper()
        if compact_text != "TABLEOFCONTENTS":
            continue
        x0, y0, x1, _y1 = line["bbox"]
        center_x = (x0 + x1) / 2
        centered = abs(center_x - page_width / 2) <= page_width * 0.15
        in_header_area = y0 <= page_height * 0.15
        visually_primary = line["is_bold"] and line["max_size"] >= max(14.0, typical_size + 4.0)
        if centered and in_header_area and visually_primary:
            return True
    return False


def find_clarivate_toc_page(doc: fitz.Document) -> fitz.Page | None:
    for footer_number in (3, 4):
        page = find_clarivate_page_by_footer(doc, footer_number)
        if page is not None and is_clarivate_table_of_contents_page(page):
            return page
    return None


def find_corsearch_toc_page(doc: fitz.Document) -> fitz.Page | None:
    footer_numbers = (1, 2, 3, 4, 5)
    for footer_number in footer_numbers:
        page = find_corsearch_page_by_footer(doc, footer_number)
        if page is not None and is_corsearch_table_of_contents_page(page):
            return page
    return None


def is_corsearch_table_of_contents_or_continued_page(page: fitz.Page) -> bool:
    lines = page_lines(page)
    sizes = [line["max_size"] for line in lines]
    typical_size = median(sizes) if sizes else 0.0
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    for line in lines:
        compact_text = re.sub(r"[^A-Za-z]+", "", line["text"]).upper()
        if not compact_text.startswith("TABLEOFCONTENTS"):
            continue
        x0, y0, x1, _y1 = line["bbox"]
        center_x = (x0 + x1) / 2
        centered = abs(center_x - page_width / 2) <= page_width * 0.2
        in_header_area = y0 <= page_height * 0.15
        visually_primary = line["is_bold"] and line["max_size"] >= max(14.0, typical_size + 4.0)
        if centered and in_header_area and visually_primary:
            return True
    return False


def find_corsearch_toc_pages(doc: fitz.Document) -> list[fitz.Page]:
    toc_pages: list[fitz.Page] = []
    footer_numbers = (1, 2, 3, 4, 5)
    for footer_number in footer_numbers:
        page = find_corsearch_page_by_footer(doc, footer_number)
        if page is not None and is_corsearch_table_of_contents_or_continued_page(page):
            toc_pages.append(page)
    return toc_pages


def find_toc_page(doc: fitz.Document) -> fitz.Page | None:
    for page_number in (3, 4, 2):
        page_index = page_number - 1
        if 0 <= page_index < doc.page_count and is_table_of_contents_page(doc[page_index]):
            return doc[page_index]
    return None


def extract_toc_page_numbers(page: fitz.Page) -> tuple[str, str]:
    lines = page_lines(page)
    state_starting_page = "error"
    state_end_page = "error"
    section = ""

    for line in lines:
        text = line["text"]
        if text == "State Trademark Report":
            section = "state"
            continue
        if text == "Web Common Law":
            section = "web"
            continue
        if line["is_bold"] and line["bbox"][0] < float(page.rect.width) * 0.25:
            section = ""
            continue

        if section == "state" and text.startswith("State Summary"):
            match = re.search(r"(\d+)\s*$", text)
            if match:
                state_starting_page = match.group(1)
            continue

        if section == "web" and text.startswith("Analyst Review"):
            match = re.search(r"(\d+)\s*$", text)
            if match:
                state_end_page = match.group(1)
                section = ""
            continue

    return state_starting_page, state_end_page


def line_same_row(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_y0, _first_y1 = first["bbox"][1], first["bbox"][3]
    second_y0, second_y1 = second["bbox"][1], second["bbox"][3]
    first_mid = (first_y0 + _first_y1) / 2
    second_mid = (second_y0 + second_y1) / 2
    return abs(first_mid - second_mid) <= max(first["max_size"], second["max_size"]) * 0.7


def right_side_value_on_same_row(lines: list[dict[str, Any]], label_line: dict[str, Any]) -> str:
    _label_x0, _label_y0, label_x1, _label_y1 = label_line["bbox"]
    values = [
        line
        for line in lines
        if line is not label_line
        and line["bbox"][0] >= label_x1
        and line_same_row(label_line, line)
    ]
    values = sorted(values, key=lambda item: item["bbox"][0])
    return normalize_text(" ".join(line["text"] for line in values))


def extract_clarivate_label_value(lines: list[dict[str, Any]], labels: tuple[str, ...]) -> str:
    for index, line in enumerate(lines):
        text = line["text"]
        label = next((candidate for candidate in labels if text.startswith(candidate)), "")
        if not label:
            continue

        inline_value = normalize_text(text[len(label) :])
        if inline_value:
            return inline_value

        row_value = right_side_value_on_same_row(lines, line)
        if row_value:
            return row_value

        values: list[str] = []
        previous_value_bottom: float | None = None
        label_x0, _label_y0, _label_x1, label_y1 = line["bbox"]
        for next_line in lines[index + 1 :]:
            next_text = next_line["text"]
            x0, y0, _x1, _y1 = next_line["bbox"]
            if y0 <= label_y1:
                continue
            if next_line["is_bold"] and x0 <= label_x0 + 5:
                break
            if any(next_text.startswith(candidate) for candidate in labels):
                break
            if previous_value_bottom is not None and y0 - previous_value_bottom > 30:
                break
            if x0 >= label_x0 - 5:
                values.append(next_text)
                previous_value_bottom = next_line["bbox"][3]
        if values:
            return "\n".join(values)

    return "error"


def number_on_same_row(lines: list[dict[str, Any]], label_line: dict[str, Any]) -> str:
    values = [
        line
        for line in lines
        if line is not label_line
        and line["bbox"][0] > label_line["bbox"][2]
        and line_same_row(label_line, line)
        and re.fullmatch(r"\d+", line["text"])
    ]
    if not values:
        inline_match = re.search(r"(\d+)\s*$", label_line["text"])
        return inline_match.group(1) if inline_match else ""
    return max(values, key=lambda item: item["bbox"][0])["text"]


def is_common_law_section_heading(line: dict[str, Any], page_width: float) -> bool:
    x0 = float(line["bbox"][0])
    return bool(
        line["is_bold"]
        and line_is_dark(line)
        and x0 < page_width * 0.25
    )


def common_law_toc_page_number(lines: list[dict[str, Any]], line: dict[str, Any]) -> str:
    page_number = number_on_same_row(lines, line)
    if page_number:
        return page_number

    match = re.search(r"(\d+)\s*$", line["text"])
    return match.group(1) if match else ""


def extract_compumark_common_law_toc_page_numbers(
    page: fitz.Page,
) -> dict[str, str]:
    lines = page_lines(page)
    page_width = float(page.rect.width)
    web_start_page = "error"
    web_end_page = "error"
    common_compumark_start = "error"
    common_compumark_end = "error"
    Business_compumark_start = "error"
    Business_compumark_end = "error"
    domain_compumark_start = "error"
    domain_compumark_end = "error"

    current_section = ""
    has_seen_state_trademark_report = False

    for line in lines:
        text = normalize_text(line["text"])

        if text == "State Trademark Report" and is_common_law_section_heading(line, page_width):
            has_seen_state_trademark_report = True
            current_section = ""
            continue

        if (
            text == "Web Common Law"
            and has_seen_state_trademark_report
            and is_common_law_section_heading(line, page_width)
        ):
            current_section = WEB_COMMON_LAW_SECTION
            continue

        if text == "Common Law Database Report" and is_common_law_section_heading(line, page_width):
            current_section = COMMON_LAW_DATABASE_SECTION
            continue

        if text == "Common Law Business Name Report" and is_common_law_section_heading(line, page_width):
            current_section = "business_name"
            continue

        if text == "Internet Domain Name Report" and is_common_law_section_heading(line, page_width):
            current_section = "domain_name"
            continue

        if text == "References" and is_common_law_section_heading(line, page_width):
            current_section = "references"
            continue

        if is_common_law_section_heading(line, page_width):
            current_section = ""
            continue

        # --- Section start/end extractions ---

        if current_section == WEB_COMMON_LAW_SECTION and text.startswith("Web Common Law Summary"):
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                web_start_page = page_number
            continue

        if (
            current_section == COMMON_LAW_DATABASE_SECTION
            and text.startswith("Common Law Database Summary")
        ):
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                web_end_page = page_number
            continue

        if (
            current_section == COMMON_LAW_DATABASE_SECTION
            and text.startswith("Common Law Database Citations")
        ):
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                common_compumark_start = page_number
            continue

        if (
            current_section == "business_name"
            and text.startswith("Common Law Business Name Summary")
        ):
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                common_compumark_end = str(int(page_number) - 1)
            continue

        if (
            current_section == "business_name"
            and text.startswith("Common Law Business Name Citations")
        ):
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                Business_compumark_start = page_number
            continue

        if (
            current_section == "domain_name"
            and text.startswith("Internet Domain Name Summary")
        ):
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                Business_compumark_end = str(int(page_number) - 1)
                domain_compumark_start = page_number
            continue

        if (
            current_section == "references"
            and text.startswith("Common Law References")
        ):
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                domain_compumark_end = str(int(page_number) - 1)
            continue

    return {
        "Web_start_page": web_start_page,
        "Web_end_page": web_end_page,
        "common_compumark_start": common_compumark_start,
        "common_compumark_end": common_compumark_end,
        "Business_compumark_start": Business_compumark_start,
        "Business_compumark_end": Business_compumark_end,
        "domain_compumark_start": domain_compumark_start,
        "domain_compumark_end": domain_compumark_end,
    }


def extract_compumark_common_law_pdf(doc: fitz.Document) -> dict[str, str]:
    toc_page = find_toc_page(doc)
    if toc_page is None:
        return {
            "Web_start_page": "error",
            "Web_end_page": "error",
            "common_compumark_start": "error",
            "common_compumark_end": "error",
            "Business_compumark_start": "error",
            "Business_compumark_end": "error",
            "domain_compumark_start": "error",
            "domain_compumark_end": "error",
        }

    return extract_compumark_common_law_toc_page_numbers(toc_page)


def extract_clarivate_toc_page_numbers(page: fitz.Page) -> tuple[str, str]:
    lines = page_lines(page)
    state_starting_page = "error"
    state_end_page = "error"
    in_us_states = False

    for line in lines:
        text = line["text"]
        if text == "US States Trademark Report":
            in_us_states = True
            continue

        if in_us_states and text == "US States Overview List":
            page_number = number_on_same_row(lines, line)
            if page_number:
                state_starting_page = page_number
            continue

        if text == "Web Common Law":
            page_number = number_on_same_row(lines, line)
            if page_number:
                state_end_page = str(int(page_number) - 1)
            break

    return state_starting_page, state_end_page


def extract_clarivate_section_ranges(
    page: fitz.Page,
) -> dict[str, str]:
    """Extract page ranges for Web Common Law, Common Law, Business Names,
    and Domain Names sections from the Clarivate Table of Contents page."""
    lines = page_lines(page)

    web_clarivate_start = "error"
    web_clarivate_end = "error"
    common_clarivate_start = "error"
    common_clarivate_end = "error"
    Business_clarivate_start = "error"
    Business_clarivate_end = "error"
    domain_clarivate_start = "error"
    domain_clarivate_end = "error"

    def get_clean_pn(lines, line):
        pn = number_on_same_row(lines, line)
        if pn:
            # Only keep numeric parts (e.g., '521' from '521 Domain Names' or just '521')
            match = re.search(r"(\d+)", pn)
            return match.group(1) if match else None
        return None

    for line in lines:
        text = line["text"]

        # --- START page extraction (Overview List rows) ---
        if text == "Web Common Law Overview List":
            pn = get_clean_pn(lines, line)
            if pn:
                web_clarivate_start = pn
            continue

        if text == "Common Law Overview List":
            pn = get_clean_pn(lines, line)
            if pn:
                common_clarivate_start = pn
            continue

        if text == "Business Names Overview List":
            pn = get_clean_pn(lines, line)
            if pn:
                Business_clarivate_start = pn
            continue

        if text == "Domain Names Overview List":
            pn = get_clean_pn(lines, line)
            if pn:
                domain_clarivate_start = pn
            continue

        # --- END page extraction (main heading rows, {number} - 1) ---
        if text == "Common Law":
            pn = get_clean_pn(lines, line)
            if pn:
                web_clarivate_end = str(int(pn) - 1)
            continue

        if text == "Business Names":
            pn = get_clean_pn(lines, line)
            if pn:
                common_clarivate_end = str(int(pn) - 1)
            continue

        if text == "Domain Names":
            pn = get_clean_pn(lines, line)
            if pn:
                Business_clarivate_end = str(int(pn) - 1)
            continue

        if text == "References":
            pn = get_clean_pn(lines, line)
            if pn:
                domain_clarivate_end = str(int(pn) - 1)
            continue

    return {
        "web_clarivate_start": web_clarivate_start,
        "web_clarivate_end": web_clarivate_end,
        "common_clarivate_start": common_clarivate_start,
        "common_clarivate_end": common_clarivate_end,
        "Business_clarivate_start": Business_clarivate_start,
        "Business_clarivate_end": Business_clarivate_end,
        "domain_clarivate_start": domain_clarivate_start,
        "domain_clarivate_end": domain_clarivate_end,
    }


def extract_corsearch_toc_page_numbers(page: fitz.Page) -> tuple[str, str]:
    lines = page_lines(page)
    state_starting_page = "error"
    state_end_page = "error"
    section = ""

    for line in lines:
        text = line["text"]
        if text == "US STATE":
            section = "us_state"
            continue
        if text == "COMMON LAW":
            section = "common_law"
            continue
        if line["is_bold"] and line["bbox"][0] < float(page.rect.width) * 0.25:
            section = ""
            continue

        if section == "us_state" and text == "State Search Results":
            page_number = number_on_same_row(lines, line)
            if page_number:
                state_starting_page = page_number
            continue

        if section == "common_law" and text == "Common Law Search Strategy":
            page_number = number_on_same_row(lines, line)
            if page_number:
                state_end_page = str(int(page_number) - 2)
            continue

    return state_starting_page, state_end_page


def is_corsearch_common_law_section_heading(line: dict[str, Any], page_width: float) -> bool:
    return bool(line["is_bold"] and line["bbox"][0] < page_width * 0.25)


def extract_corsearch_common_law_toc_page_numbers(page: fitz.Page) -> tuple[str, str]:
    lines = page_lines(page)
    page_width = float(page.rect.width)
    Web_corsearch_start_page = "error"
    Web_corsearch_end_page = "error"
    section = ""

    for line in lines:
        text = normalize_text(line["text"])

        if text == "COMMON LAW" and is_corsearch_common_law_section_heading(line, page_width):
            section = "common_law"
            continue

        if text == "BUSINESS NAME" and is_corsearch_common_law_section_heading(line, page_width):
            section = "business_name"
            continue

        if is_corsearch_common_law_section_heading(line, page_width):
            section = ""
            continue

        if section == "common_law" and text == "Common Law Summary":
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                Web_corsearch_start_page = page_number
            continue

        if section == "business_name" and text == "Business Name Information":
            page_number = common_law_toc_page_number(lines, line)
            if page_number:
                Web_corsearch_end_page = str(int(page_number) - 1)
            continue

    return Web_corsearch_start_page, Web_corsearch_end_page


def extract_corsearch_common_law_pdf(doc: fitz.Document) -> dict[str, str]:
    toc_page = find_corsearch_toc_page(doc)
    if toc_page is None:
        Web_corsearch_start_page = "error"
        Web_corsearch_end_page = "error"
    else:
        Web_corsearch_start_page, Web_corsearch_end_page = extract_corsearch_common_law_toc_page_numbers(toc_page)

    return {
        "Web_corsearch_start_page": Web_corsearch_start_page,
        "Web_corsearch_end_page": Web_corsearch_end_page,
    }


def extract_corsearch_business_domain_toc_page_numbers(
    pages: list[fitz.Page],
) -> tuple[str, str, str, str]:
    business_corsearch_start = "error"
    business_corsearch_end = "error"
    domain_corsearch_start = "error"
    domain_corsearch_end = "error"
    section = ""

    for page in pages:
        lines = page_lines(page)
        page_width = float(page.rect.width)

        for line in lines:
            text = normalize_text(line["text"])

            if text == "BUSINESS NAME" and is_corsearch_common_law_section_heading(line, page_width):
                section = "business_name"
                continue

            if text == "DOMAIN NAME" and is_corsearch_common_law_section_heading(line, page_width):
                section = "domain_name"
                continue

            if text in {"WEB RESULTS", "WEB RESULT"} and is_corsearch_common_law_section_heading(line, page_width):
                section = "web_results"
                continue

            if is_corsearch_common_law_section_heading(line, page_width):
                section = ""
                continue

            if section == "business_name" and text == "Business Name Summary":
                page_number = common_law_toc_page_number(lines, line)
                if page_number:
                    business_corsearch_start = page_number
                continue

            if section == "domain_name" and text == "Domain Name Search Strategy":
                page_number = common_law_toc_page_number(lines, line)
                if page_number:
                    business_corsearch_end = str(int(page_number) - 1)
                continue

            if section == "domain_name" and text == "Domain Name Summary":
                page_number = common_law_toc_page_number(lines, line)
                if page_number:
                    domain_corsearch_start = page_number
                continue

            if section == "web_results" and text == "Web Results Search Strategy":
                page_number = common_law_toc_page_number(lines, line)
                if page_number:
                    domain_corsearch_end = str(int(page_number) - 1)
                continue

    return business_corsearch_start, business_corsearch_end, domain_corsearch_start, domain_corsearch_end


def extract_corsearch_business_domain_pdf(doc: fitz.Document) -> dict[str, str]:
    toc_pages = find_corsearch_toc_pages(doc)
    if not toc_pages:
        business_corsearch_start = "error"
        business_corsearch_end = "error"
        domain_corsearch_start = "error"
        domain_corsearch_end = "error"
    else:
        (
            business_corsearch_start,
            business_corsearch_end,
            domain_corsearch_start,
            domain_corsearch_end,
        ) = extract_corsearch_business_domain_toc_page_numbers(toc_pages)

    return {
        "business_corsearch_start": business_corsearch_start,
        "business_corsearch_end": business_corsearch_end,
        "domain_corsearch_start": domain_corsearch_start,
        "domain_corsearch_end": domain_corsearch_end,
    }


def extract_corsearch_web_results_toc_page_numbers(
    pages: list[fitz.Page],
) -> tuple[str, str]:
    Webresult_corsearch_start = "error"
    Webresult_corsearch_end = "error"
    section = ""

    for page in pages:
        lines = page_lines(page)
        page_width = float(page.rect.width)

        for line in lines:
            text = normalize_text(line["text"])

            if text in {"WEB RESULTS", "WEB RESULT"} and is_corsearch_common_law_section_heading(line, page_width):
                section = "web_results"
                continue

            if text == "APPENDIX" and is_corsearch_common_law_section_heading(line, page_width):
                section = "appendix"
                continue

            if is_corsearch_common_law_section_heading(line, page_width):
                section = ""
                continue

            if section == "web_results" and text == "Web Results Summary":
                page_number = common_law_toc_page_number(lines, line)
                if page_number:
                    Webresult_corsearch_start = page_number
                continue

            if section == "appendix" and text == "Federal Classifier":
                page_number = common_law_toc_page_number(lines, line)
                if page_number:
                    Webresult_corsearch_end = str(int(page_number) - 1)
                continue

    return Webresult_corsearch_start, Webresult_corsearch_end


def extract_corsearch_web_results_pdf(doc: fitz.Document) -> dict[str, str]:
    toc_pages = find_corsearch_toc_pages(doc)
    if not toc_pages:
        Webresult_corsearch_start = "error"
        Webresult_corsearch_end = "error"
    else:
        Webresult_corsearch_start, Webresult_corsearch_end = extract_corsearch_web_results_toc_page_numbers(toc_pages)

    return {
        "Webresult_corsearch_start": Webresult_corsearch_start,
        "Webresult_corsearch_end": Webresult_corsearch_end,
    }


def extract_clarivate_pdf(doc: fitz.Document, vendor_name: str) -> dict[str, str]:
    result = dict(CLARIVATE_OUTPUT_KEYS)
    result["vender_name"] = "Clarivate" if is_clarivate_vendor_name(vendor_name) else vendor_name

    if doc.page_count > 1:
        detail_lines = page_lines(doc[1])
        result["Mark Searched"] = extract_clarivate_label_value(detail_lines, ("Mark Searched:",))
        result["Classes Searched"] = extract_clarivate_label_value(detail_lines, ("Classes Searched:",))
        result["Goods/Services Searched"] = extract_clarivate_label_value(
            detail_lines,
            ("Goods and Services:", "Goods/Services:"),
        )

    toc_page = find_clarivate_toc_page(doc)
    if toc_page is not None:
        state_starting_page, state_end_page = extract_clarivate_toc_page_numbers(toc_page)
        result["state_starting_page"] = state_starting_page
        result["state_end_page"] = state_end_page
        result.update(extract_clarivate_section_ranges(toc_page))

    return result


def extract_corsearch_pdf(doc: fitz.Document, vendor_name: str) -> dict[str, str]:
    result = dict(CORSEARCH_OUTPUT_KEYS)
    result.update(extract_corsearch_cover_page(doc[0], "Corsearch" if is_corsearch_vendor_name(vendor_name) else vendor_name))

    toc_page = find_corsearch_toc_page(doc)
    if toc_page is not None:
        state_starting_page, state_end_page = extract_corsearch_toc_page_numbers(toc_page)
        result["state_starting_page"] = state_starting_page
        result["state_end_page"] = state_end_page

    return result


def timestamped_output_path(pdf_path: str) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(pdf_path).stem).strip("._-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_stem or 'layer1'}_layer1_{timestamp}.json"


def timestamped_common_law_output_path(pdf_path: str) -> str:
    output_dir = Path("json+image_compumark_var2")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(pdf_path).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_stem).strip("._-") or "common_law"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(output_dir / f"{safe_stem}_common_law_{timestamp}.json")


def serialize_json_payload(data: dict[str, Any]) -> str:
    return orjson.dumps(
        data,
        option=orjson.OPT_INDENT_2,
    ).decode("utf-8")


def extract_pdf(pdf_path: str, context: DocumentContext | None = None) -> dict[str, str]:
    result = dict(OUTPUT_KEYS)
    doc = context.get_fitz_doc() if context else fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return result

        # --- NEW FOVEA DETECTION BLOCK ---
        if is_fovea_vendor(doc[0]):
            result["vendor_name"] = VENDOR_FOVEA
            return result
        # ---------------------------------

        clarivate_vendor_name = extract_clarivate_vendor_name(doc[0])
        if is_clarivate_vendor_name(clarivate_vendor_name):
            return extract_clarivate_pdf(doc, clarivate_vendor_name)

        corsearch_vendor_name = extract_corsearch_vendor_name(doc[0])
        if is_corsearch_vendor_name(corsearch_vendor_name):
            return extract_corsearch_pdf(doc, corsearch_vendor_name)

        result.update(extract_cover_page(doc[0]))
        toc_page = find_toc_page(doc)
        if toc_page is not None:
            state_starting_page, state_end_page = extract_toc_page_numbers(toc_page)
            result["state_starting_page"] = state_starting_page
            result["state_end_page"] = state_end_page
    finally:
        if not context:
            doc.close()

    return result


def detect_vendor_type(layer1_data: dict[str, Any]) -> str:
    vendor_name = str(
        layer1_data.get("vendor_name")
        or layer1_data.get("vender_name")
        or ""
    )
    # --- NEW FOVEA CHECK ---
    if vendor_name == VENDOR_FOVEA:
        return VENDOR_FOVEA
    # -----------------------
    
    if is_clarivate_vendor_name(vendor_name):
        return VENDOR_CLARIVATE
    if is_corsearch_vendor_name(vendor_name):
        return VENDOR_CORSEARCH
    if normalize_vendor_for_comparison(vendor_name) == "compumark":
        return VENDOR_COMPUMARK
    return "error"


def import_clarivate_extractor() -> Any:
    try:
        from . import state_clarivate
        return state_clarivate
    except (ImportError, ValueError):
        try:
            import state_clarivate
            return state_clarivate
        except ImportError:
            try:
                return importlib.import_module("variation1")
            except ModuleNotFoundError:
                pass
            raise ImportError("Could not find state_clarivate or variation1 extractor.")


def dispatch_vendor_extraction(pdf_path: str, vendor_type: str | None = None, context: DocumentContext | None = None) -> dict[str, Any]:
    if vendor_type is None:
        vendor_type = detect_vendor_type(extract_pdf(pdf_path, context=context))

    # --- NEW FOVEA DISPATCH ---
    if vendor_type == VENDOR_FOVEA:
        try:
            from .fovea_Docx import fove_pdf_extract
        except (ImportError, ValueError):
            from fovea_Docx import fove_pdf_extract
        return fove_pdf_extract.extract_pdf(pdf_path)
    # --------------------------

    if vendor_type == VENDOR_COMPUMARK:
        try:
            from . import variation_extraction_compumark
        except (ImportError, ValueError):
            import variation_extraction_compumark

        return variation_extraction_compumark.extract_pdf(pdf_path)
    if vendor_type == VENDOR_CLARIVATE:
        variation1 = import_clarivate_extractor()

        return variation1.extract_state_summary(pdf_path)
    if vendor_type == VENDOR_CORSEARCH:
        try:
            from . import state_corsearch
        except (ImportError, ValueError):
            import state_corsearch

        return state_corsearch.extract_corsearch_state_summary(pdf_path, context=context)

    raise RuntimeError(f"Unsupported or undetected vendor for PDF: {pdf_path}")


def dispatch_common_law_extraction(
    pdf_path: str,
    vendor_type: str | None = None,
    context: DocumentContext | None = None,
) -> dict[str, Any]:
    if vendor_type is None:
        vendor_type = detect_vendor_type(extract_pdf(pdf_path, context=context))

    if vendor_type == VENDOR_COMPUMARK:
        with fitz.open(pdf_path) as doc:
            page_range = extract_compumark_common_law_pdf(doc)
        try:
            from .common_law import cl_compumark
        except (ImportError, ValueError):
            from common_law import cl_compumark

        return cl_compumark.extract_pdf(
            pdf_path,
            page_range["Web_start_page"],
            page_range["Web_end_page"],
            common_compumark_start=page_range.get("common_compumark_start"),
            common_compumark_end=page_range.get("common_compumark_end"),
            Business_compumark_start=page_range.get("Business_compumark_start"),
            Business_compumark_end=page_range.get("Business_compumark_end"),
            domain_compumark_start=page_range.get("domain_compumark_start"),
            domain_compumark_end=page_range.get("domain_compumark_end"),
        )
    if vendor_type == VENDOR_CLARIVATE:
        # 1. Trigger the new traditional common law database extractor first.
        # This processes multi-page tables and writes standalone data into JSONcommon_database_clarivate/
        try:
            try:
                from . import Only_commonLaw_Clarivate
            except (ImportError, ValueError):
                import Only_commonLaw_Clarivate
            Only_commonLaw_Clarivate.extract_pdf(pdf_path)
        except Exception as exc:
            print(f"[{RETURN_TYPE_COMMON_LAW}] Standalone Traditional Database extraction failed: {exc}")

        # 2. Proceed to run the existing web common law extractor second.
        # This executes the parallel LLM web-browsing search tools and returns its main data block to corsearch_result/
        try:
            from . import commonLaw_clarivate
        except (ImportError, ValueError):
            import commonLaw_clarivate
        return commonLaw_clarivate.extract_pdf(pdf_path)
    if vendor_type == VENDOR_CORSEARCH:
        try:
            from .common_law import main as cl_main
        except (ImportError, ValueError):
            from common_law import main as cl_main
        data = cl_main.extract_pdf(pdf_path, context=context)
        return cl_main.web_results_output_payload(data)

    raise RuntimeError(f"Unsupported or undetected vendor for common law PDF: {pdf_path}")


def import_corsearch_bnd_extractor() -> Any:
    try:
        from .business_domain import Business_and_Domain_Corsearch
    except (ImportError, ValueError):
        from business_domain import Business_and_Domain_Corsearch
    return Business_and_Domain_Corsearch


def dispatch_business_domain_extraction(
    pdf_path: str,
    vendor_type: str | None = None,
    context: DocumentContext | None = None,
) -> dict[str, Any]:
    if vendor_type is None:
        vendor_type = detect_vendor_type(extract_pdf(pdf_path, context=context))

    if vendor_type == VENDOR_CLARIVATE:
        if context and context.clarivate_ranges:
            ranges = context.clarivate_ranges
        else:
            with fitz.open(pdf_path) as doc:
                toc_page = find_clarivate_toc_page(doc)
                if toc_page is None:
                    return {"return_type": "B_&_D", "vendor_name": "Clarivate", "status": "toc_not_found"}
                ranges = extract_clarivate_section_ranges(toc_page)
            
        try:
            from . import Business_clarivate
        except (ImportError, ValueError):
            import Business_clarivate
        return Business_clarivate.extract_pdf(
            pdf_path, 
            business_start=ranges.get("Business_clarivate_start"), 
            business_end=ranges.get("Business_clarivate_end"),
            context=context
        )

    if vendor_type == VENDOR_COMPUMARK:
        with fitz.open(pdf_path) as doc:
            ranges = extract_compumark_common_law_pdf(doc)
            
        try:
            from . import domaincompumark
        except (ImportError, ValueError):
            import domaincompumark
        start = ranges.get("domain_compumark_start")
        end = ranges.get("domain_compumark_end")
        return domaincompumark.extract_pdf(
            pdf_path, 
            domain_compumark_start=start,
            domain_compumark_end=end
        )

    if vendor_type == VENDOR_CORSEARCH:
        module = import_corsearch_bnd_extractor()
        return module.extract_corsearch_business_domain(pdf_path, context=context)

    raise RuntimeError(f"Unsupported or undetected vendor for business/domain PDF: {pdf_path}")


def import_cl_corsearch():
    """Import cl_corsearch from the common_law sub-directory."""
    try:
        from .common_law import cl_corsearch
    except (ImportError, ValueError):
        from common_law import cl_corsearch
    return cl_corsearch


def dispatch_cml_corsearch_extraction(
    pdf_path: str,
    vendor_type: str | None = None,
    context: DocumentContext | None = None,
) -> dict[str, Any]:
    if vendor_type is None:
        vendor_type = detect_vendor_type(extract_pdf(pdf_path, context=context))
    if vendor_type == VENDOR_CLARIVATE:
        if context and context.clarivate_ranges:
            ranges = context.clarivate_ranges
        else:
            with fitz.open(pdf_path) as doc:
                toc_page = find_clarivate_toc_page(doc)
                if toc_page is None:
                    return {"Identical Names": [], "Similar Names": [], "status": "toc_not_found"}
                ranges = extract_clarivate_section_ranges(toc_page)
            
        try:
            from . import domainclarivate
        except (ImportError, ValueError):
            import domainclarivate
        return domainclarivate.extract_pdf(
            pdf_path, 
            domain_start=ranges.get("domain_clarivate_start"),
            domain_end=ranges.get("domain_clarivate_end"),
            context=context
        )

    if vendor_type == VENDOR_CORSEARCH:
        module = import_cl_corsearch()
        return module.extract_pdf(pdf_path, context=context)
    if vendor_type == VENDOR_COMPUMARK:
        with fitz.open(pdf_path) as doc:
            page_range = extract_compumark_common_law_pdf(doc)
        try:
            from . import compumark_commonLaw
        except (ImportError, ValueError):
            import compumark_commonLaw
        try:
            from . import compumark_business
        except (ImportError, ValueError):
            import compumark_business
        
        # Sequential Extraction
        # 1. Common Law Database
        cl_result = compumark_commonLaw.extract_pdf(
            pdf_path,
            page_range.get("common_compumark_start"),
            page_range.get("common_compumark_end"),
        )
        
        # 2. Business Name
        bus_result = compumark_business.extract_bus_records(
            pdf_path,
            page_range.get("Business_compumark_start"),
            page_range.get("Business_compumark_end"),
        )
        
        return cl_result
    raise RuntimeError(f"cl_corsearch branch not supported for vendor: {vendor_type}")



def dispatched_output_path(vendor_type: str, pdf_path: str, requested_output: str | None, branch_type: str) -> str:
    if requested_output:
        base_path = Path(requested_output)
        return str(base_path.parent / f"{branch_type}_{base_path.name}")

    if branch_type == RETURN_TYPE_COMMON_LAW and vendor_type == VENDOR_COMPUMARK:
        output_path = timestamped_common_law_output_path(pdf_path)
    elif vendor_type == VENDOR_COMPUMARK:
        import variation_extraction_compumark
        output_path = variation_extraction_compumark.timestamped_output_path_in_json_image_folder(pdf_path)
    elif vendor_type == VENDOR_CLARIVATE:
        output_dir = Path("corsearch_result")
        output_dir.mkdir(parents=True, exist_ok=True)
        if branch_type == RETURN_TYPE_COMMON_LAW:
            output_path = str(output_dir / f"clarivate_common_law_{Path(pdf_path).stem}.json")
        elif branch_type == RETURN_TYPE_BUSINESS_DOMAIN:
            output_path = str(output_dir / f"clarivate_business_domain_{Path(pdf_path).stem}.json")
        elif branch_type == RETURN_TYPE_CML_CORSEARCH:
            output_path = str(output_dir / f"clarivate_domain_names_{Path(pdf_path).stem}.json")
        else:
            variation1 = import_clarivate_extractor()
            # Ensure the task_id (from pdf_path stem) is in the filename for app.py detection
            stem = Path(pdf_path).stem
            output_path = str(output_dir / f"{stem}_state_summary.json")
        
    elif vendor_type == VENDOR_CORSEARCH and branch_type == RETURN_TYPE_CML_CORSEARCH:
        # cl_corsearch uses its own timestamped name under corsearch_result/
        stem = Path(pdf_path).stem.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("corsearch_result")
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{stem}_cml_{timestamp}.json")
    elif vendor_type == VENDOR_CORSEARCH:
        try:
            from . import state_corsearch
        except (ImportError, ValueError):
            import state_corsearch
        output_path = str(state_corsearch.timestamped_output_path(pdf_path))
    else:
        output_path = timestamped_output_path(pdf_path)

    path_obj = Path(output_path)
    return str(path_obj.parent / f"{branch_type}_{path_obj.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect report vendor and dispatch to the matching state extractor.")
    parser.add_argument("pdf_path", nargs="?", default=DEFAULT_PDF_PATH, help="Input PDF path.")
    parser.add_argument("-o", "--output", help="Optional output JSON path. Defaults to timestamped filename.")
    return parser.parse_args()


def ensure_modules_initialized(vendor_type: str) -> None:
    """Thread-safe singleton initialization to prevent concurrent import race conditions.
    This solves the 'partially initialized module' error during parallel thread startup."""
    global _MODULES_INITIALIZED
    with _INITIALIZATION_LOCK:
        if _MODULES_INITIALIZED:
            return
        
        print(f"[INITIALIZATION] Pre-loading modules for {vendor_type}...")
        try:
            # First, stabilize core packages that often suffer from race conditions
            import openai
            import instructor
            # Accessing a nested type often triggers the full initialization of members
            getattr(openai, "AzureOpenAI", None)
            getattr(instructor, "Mode", None)
        except Exception as e:
            print(f"[INITIALIZATION] Error stabilizing core packages: {e}")

        try:
            # Then, pre-import vendor specific modules
            if vendor_type == VENDOR_COMPUMARK:
                pass
            elif vendor_type == VENDOR_CORSEARCH:
                pass
            elif vendor_type == VENDOR_CLARIVATE:
                pass
            
            _MODULES_INITIALIZED = True
            print("[INITIALIZATION] Modules initialized.")
        except Exception as e:
            print(f"[INITIALIZATION] Warning: Some vendor modules failed to pre-load (this may be normal): {e}")
            _MODULES_INITIALIZED = True # Still mark as initialized to avoid repeat attempts


def run_extraction_branch(pdf_path: str, vendor_type: str, branch_type: str, requested_output: str | None, context: DocumentContext | None = None) -> None:
    # Guardrail checking for Clarivate to ensure Corsearch logic is completely untouched
    if vendor_type == VENDOR_CLARIVATE:
        # We now support all branches for Clarivate: state, common_law, B&D, and domain (cml_corsearch)
        pass

    print(f"[{branch_type}] Starting extraction for Vendor: {vendor_type} (context={context})")
    try:
        if branch_type == RETURN_TYPE_COMMON_LAW:
            data = dispatch_common_law_extraction(pdf_path, vendor_type, context=context)
        elif branch_type == RETURN_TYPE_BUSINESS_DOMAIN:
            data = dispatch_business_domain_extraction(pdf_path, vendor_type, context=context)
        elif branch_type == RETURN_TYPE_CML_CORSEARCH:
            data = dispatch_cml_corsearch_extraction(pdf_path, vendor_type, context=context)
        else:
            data = dispatch_vendor_extraction(pdf_path, vendor_type, context=context)
            
        payload = serialize_json_payload(data)
        output_path = dispatched_output_path(vendor_type, pdf_path, requested_output, branch_type)
        output_parent = Path(output_path).parent
        if str(output_parent) not in {"", "."}:
            output_parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        print(f"[{branch_type}] Saved to {output_path}")
    except Exception as exc:
        print(f"[{branch_type}] Failed with error: {exc}")


def build_document_context(pdf_path: str, doc: fitz.Document, pdf_bytes: bytes, vendor_type: str) -> DocumentContext:
    ctx = DocumentContext(pdf_path=pdf_path, pdf_bytes=pdf_bytes, vendor_type=vendor_type)
    if vendor_type == VENDOR_CLARIVATE:
        toc_page = find_clarivate_toc_page(doc)
        if toc_page:
            ctx.clarivate_ranges = extract_clarivate_section_ranges(toc_page)
            state_start, state_end = extract_clarivate_toc_page_numbers(toc_page)
            ctx.clarivate_ranges["state_starting_page"] = state_start
            ctx.clarivate_ranges["state_end_page"] = state_end
    return ctx

def main() -> None:
    require_conda_env()
    args = parse_args()
    import json
    
    file_path = args.pdf_path
    if not validate_input_file(file_path):
        return

    file_ext = os.path.splitext(file_path)[1].lower()
    # --- NEW DOCX INTERCEPTION & ROUTING BLOCK ---
    if file_ext == ".docx":
        try:
            try:
                from .fovea_Docx import applicant_data_docx_extractor
            except (ImportError, ValueError):
                from fovea_Docx import applicant_data_docx_extractor
            import docx
        except (ImportError, ValueError):
            # 1. Dynamically route to the fovea_Docx directory
            docx_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fovea_Docx")
            if docx_module_path not in sys.path:
                sys.path.insert(0, docx_module_path)
            import applicant_data_docx_extractor
            import docx
            
        try:
            # 2. Table-based Vendor Detection (Uses untouched logic from the sub-controller)
            applicant_data_docx_extractor.validate_docx_archive(file_path)
            doc_docx = docx.Document(file_path)
            applicant_data_docx_extractor.validate_template(doc_docx)
            
            print(f"[{RETURN_TYPE_STATE_LAW}] Starting extraction for Vendor: Fovea (DOCX)")
            # 3. Execute Extraction
            result = applicant_data_docx_extractor.extract_docx(file_path)
            
            # 4. Implement 'footer_text' and 'start_extraction' at the router level
            # Extract footer text using python-docx without modifying the extractor script
            footer_text = ""
            for section in doc_docx.sections:
                for footer_def in [section.first_page_footer, section.footer, section.even_page_footer]:
                    if footer_def and not footer_def.is_linked_to_previous:
                        for p in footer_def.paragraphs:
                            if p.text.strip():
                                footer_text += p.text + " "
            
            # Inject the fields into the final payload
            result["footer_text"] = re.sub(r"\s+", " ", footer_text).strip()
            result["start_extraction"] = True
            
            # 5. Save the output and exit safely
            out_path = dispatched_output_path(VENDOR_FOVEA, file_path, args.output, RETURN_TYPE_STATE_LAW)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                
            print(f"[{RETURN_TYPE_STATE_LAW}] Saved to {out_path}")
            return # Exit the script here to bypass the PDF pipeline entirely
            
        except Exception:
            print("[VALIDATION] Unsupported DOCX template or vendor detection failed.")
            # print(f"DOCX Detail: {e}")
            print("[VALIDATION] Extraction aborted.")
            return
    # -------------

    layer1_data = extract_pdf(args.pdf_path)
    vendor_type = detect_vendor_type(layer1_data)
    
    if vendor_type == "error":
        print(f"[VALIDATION] No supported vendor detected for PDF: {args.pdf_path}")
        print("[VALIDATION] Supported vendors: CompuMark, Clarivate, Corsearch, Fovea")
        print("[VALIDATION] Extraction aborted.")
        return

    branches = [
        RETURN_TYPE_STATE_LAW,
        RETURN_TYPE_COMMON_LAW,
        RETURN_TYPE_BUSINESS_DOMAIN,
        RETURN_TYPE_CML_CORSEARCH,
    ]

    # --- NEW FOVEA THREAD OPTIMIZATION ---
    if vendor_type == VENDOR_FOVEA:
        branches = [RETURN_TYPE_STATE_LAW]
    # -------------------------------------

    # Read once into memory to share across threads
    with open(args.pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        context = build_document_context(args.pdf_path, doc, pdf_bytes, vendor_type)

    # Ensure modules are initialized in a single thread before launching pool
    ensure_modules_initialized(vendor_type)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_extraction_branch, args.pdf_path, vendor_type, branch, args.output, context): branch
                for branch in branches
            }
            
            for future in concurrent.futures.as_completed(futures):
                branch = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Branch {branch} failed with unexpected pool error: {e}")


if __name__ == "__main__":
    main()
