import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


class LegalTextParser:
    """Parser for cleaning and structuring legal opinion text.
    Cleans noisy CourtListener plain_text responses, rebuilds paragraphs,
    and extracts lightweight section boundaries that chunking can respect.
    """

    SECTION_KEYWORDS = {"BACKGROUND", "INTRODUCTION", "FACTS", "FACTUAL", "DISCUSSION", "ANALYSIS", "CONCLUSION", "JURISDICTION", "PROCEDURAL HISTORY",
        "STANDARD OF REVIEW", "LEGAL STANDARD", "HOLDING", "HOLDINGS", "ORDER", "JUDGMENT", "DECISION", "FINDINGS", "RULING", "OPINION",
        "SYLLABUS", "SUMMARY", "ISSUE", "ISSUES", "QUESTION PRESENTED", "QUESTIONS PRESENTED", "ARGUMENT", "ARGUMENTS",
    }

    NOISE_PATTERNS = [
        r"^\s*_{2,}\s*$",
        r"^\s*-{2,}\s*$",
        r"^\s*={2,}\s*$",
        r"^\s*\*{3,}\s*$",
        r"^\s*(?:\*\s*){3,}\s*$",
        r"^\s*\f\s*$",
        r"^\s*\(\s*\)\s*$",
        r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
        r"^\s*Page\s+\d+\s*$",
        r"^\s*Case:\s.*Document:.*Page:.*Date Filed:.*$",
        r"^\s*USDC\s+No\.\s*.*$",
        r"^\s*No\.\s*\d{1,6}.*$",
        r"^\s*\d{1,3}\s*$",
        r"^\s*[A-Z]{1,3}-[A-Z0-9]{3,}-\d{2,4}\s*$",
        r"^\s*-\s*\d+\s*-\s*$",
        r"^\s*â€”\s*\d+\s*â€”\s*$",
        r"^\s*/s/\s+.*$",
        r"^\s*Filed\s*$",
        r"^\s*FILED\s*$",
        r"^\s*Clerk\s*$",
        r"^\s*Before\s+the\s+Court.*$",
    ]

    CAPTION_KEYWORDS = {"plaintiff", "defendant", "appellant", "appellee", "respondent"}

    INLINE_NOISE_PATTERNS = [
        r'\s*_{3,}\s*',
        r'\s*-{3,}\s*',
        r'\s*={3,}\s*',
        r'\s*\*{3,}\s*',
    ]

    METADATA_PATTERNS = {
        'case_number': re.compile(r'Case:\s*([\d-]+)', re.IGNORECASE),
        'document_number': re.compile(r'Document:\s*([\d-]+)', re.IGNORECASE),
        'page_number': re.compile(r'Page:\s*(\d+)', re.IGNORECASE),
        'date_filed': re.compile(r'Date\s+Filed:\s*([\d/]+)', re.IGNORECASE),
        'docket_number': re.compile(r'(?:USDC\s+)?No\.\s*([\d:-]+)', re.IGNORECASE),
    }

    DIALOGUE_PATTERNS = [
        re.compile(r'^(MR\.|MS\.|MRS\.|THE\s+COURT|JUDGE|ATTORNEY|COUNSEL)\s*[A-Z\s]+:', re.IGNORECASE),
        re.compile(r'^(PLAINTIFF|DEFENDANT|WITNESS)\s*:', re.IGNORECASE),
    ]

    def __init__(self):
        self._noise_re = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]
        self._inline_noise_re = [re.compile(p) for p in self.INLINE_NOISE_PATTERNS]

    def extract_metadata(self, raw_text: str) -> Dict[str, Optional[str]]:
        """Extract metadata from raw text before cleaning."""
        metadata: Dict[str, Optional[str]] = {}
        
        for key, pattern in self.METADATA_PATTERNS.items():
            match = pattern.search(raw_text)
            metadata[key] = match.group(1) if match else None
        
        return metadata

    def _is_likely_header_line(self, line: str) -> bool:
        """Detect if a line is likely part of document header metadata."""
        lower = line.lower()
        
        if any(term in lower for term in ['court of appeals', 'district court', 'supreme court']):
            return True
        
        if any(term in lower for term in ['filed', 'summary calendar', 'clerk']):
            return True
        
        if len(line.split()) <= 5 and re.search(r'\d{4}', line):
            return True
            
        return False

    def _strip_nonprinting_chars(self, text: str) -> str:
        """Remove or normalize non-printing/OCR artifacts while preserving newlines."""
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")
        text = text.replace("\ufffd", "")

        cleaned_chars: List[str] = []
        for ch in text:
            if ch in ("\n", "\t"):
                cleaned_chars.append(ch)
                continue
            if unicodedata.category(ch).startswith("C"):
                continue
            cleaned_chars.append(ch)
        return "".join(cleaned_chars)

    def _is_caption_noise_line(self, line: str) -> bool:
        """Detect boilerplate caption/cover-page lines near the top of opinions."""
        text = (line or "").strip()
        if not text:
            return False

        if re.match(r"^(?:[IVXLCDM]+|[A-Z]|\d+)\.\s+\S+", text):
            return False

        lower = text.lower()

        if "revision until final publication" in lower:
            return True
        if lower.startswith("if this opinion indicates"):
            return True

        if "non-precedential decision" in lower:
            return True
        if "superior court o.p." in lower or "o.p. 65.37" in lower:
            return True
        if re.match(r"^(memorandum|opinion)\b", lower) and "filed" in lower:
            return True
        if lower.startswith("before:"):
            return True
        if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", lower):
            return True
        if lower.endswith(" circuit court") or "court of common pleas" in lower:
            return True

        if text.count(":") >= 4:
            return True

        if lower.startswith("appeal from the "):
            return True
        if lower.startswith("*") and "retired" in lower:
            return True

        if lower == "v" or lower == "v." or " v " in lower or " vs. " in lower:
            return True
        if re.search(r"\blc\s+no\.\b", lower):
            return True
        if re.search(r"\bnos?\.\s*\d", lower):
            return True
        if "unpublished" in lower or "for publication" in lower:
            return True

        if any(term in lower for term in self.CAPTION_KEYWORDS):
            if "/" in text or text.endswith(",") or "counter" in lower:
                return True

        if text == text.upper() and 1 <= len(text.split()) <= 8:
            if any(token in lower for token in ("state of", "court of", "before:", "clerk")):
                return True

        letters = [c for c in text if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.85 and len(text.split()) <= 14:
                if text.endswith(",") or re.search(r"\bnos?\.\s*\d", lower) or re.search(r"\blc\s+no\.\b", lower):
                    return True
                if re.search(r"\b(19|20)\d{2}\b", lower):
                    return True

        return False

    def _caption_window_should_end(self, line: str) -> bool:
        """Determine if we've reached the end of caption/header section."""
        if not line:
            return False
        upper = line.upper().strip()
        if upper.startswith("PER CURIAM"):
            return True
        if re.match(r"^(MEMORANDUM|OPINION|DECISION|ORDER)\b", line, flags=re.IGNORECASE):
            return True
        if re.match(r"^(I+V?|V?I{1,3})\.\s+", line):
            return True
        if re.search(r"\b(appeals|appealed|argues|contends|we affirm|we reverse|we vacate)\b", line, flags=re.IGNORECASE):
            if any(c.islower() for c in line) and len(line.split()) >= 6:
                return True
        return False

    def _clean_inline_noise(self, text: str) -> str:
        """Remove inline noise patterns that appear within merged paragraphs."""
        cleaned = text
        for pattern in self._inline_noise_re:
            cleaned = pattern.sub(' ', cleaned)
        
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _is_dialogue_line(self, line: str) -> bool:
        """Detect if a line is part of court transcript/dialogue."""
        for pattern in self.DIALOGUE_PATTERNS:
            if pattern.match(line.strip()):
                return True
        return False

    def clean_text(self, raw_text: str) -> str:
        """Clean and normalize raw plain text from CourtListener."""
        if not raw_text:
            return ""

        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._strip_nonprinting_chars(text)
        text = re.sub(r"\f", "\n\n", text)
        lines = text.split("\n")
        filtered_lines: List[str] = []
        header_section: List[str] = []
        body_started = False
        caption_window = True
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered_lines.append("")
                continue

            if self._is_noise_line(stripped):
                continue

            if caption_window and self._is_caption_noise_line(stripped):
                if self._caption_window_should_end(stripped):
                    caption_window = False
                continue
            if caption_window and self._caption_window_should_end(stripped):
                caption_window = False

            if not body_started and self._is_likely_header_line(stripped):
                header_section.append(stripped)
                continue
            else:
                if stripped and not self._is_likely_header_line(stripped):
                    body_started = True
            
            filtered_lines.append(stripped)

        if header_section:
            filtered_lines = header_section + [""] + filtered_lines

        normalized_lines: List[str] = []
        for line in filtered_lines:
            if line == "":
                if normalized_lines and normalized_lines[-1] == "":
                    continue
                normalized_lines.append("")
                continue
            normalized_lines.append(line)

        paragraphs = self._merge_lines_into_paragraphs(normalized_lines)
        paragraphs = self._split_inline_headers(paragraphs)
        paragraphs = [self._clean_inline_noise(p) for p in paragraphs if p]
        cleaned = "\n\n".join(paragraphs).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _split_inline_headers(self, paragraphs: List[str]) -> List[str]:
        """Split paragraphs that have section headers embedded within them.
        
        Example: "...end of previous section. A. New Section Title The content..."
        Should become: ["...end of previous section.", "A. New Section Title", "The content..."]
        """
        result: List[str] = []
        
        # Patterns for detecting inline headers
        header_patterns = [
            (re.compile(r'\.\s+([IVX]+)\.\s+([A-Z][A-Za-z\s]{3,40})(?=\s+[A-Z])'), 1),  # Roman numerals
            (re.compile(r'\.\s+([A-Z])\.\s+([A-Z][A-Za-z\s]{3,40})(?=\s+[A-Z])'), 1),   # Alpha headers
            (re.compile(r'\.\s+(\d+)\.\s+([A-Z][A-Za-z\s]{3,40})(?=\s+[A-Z])'), 1),     # Numeric headers
        ]
        
        for para in paragraphs:
            if not para or len(para) < 100:  # Too short to have inline headers
                result.append(para)
                continue
            
            split_occurred = False
            for pattern, min_offset in header_patterns:
                matches = list(pattern.finditer(para))
                if not matches:
                    continue
                
                # Split at the first match
                match = matches[0]
                # Check if this looks like a real header (has section keywords or proper structure)
                header_text = match.group(0).strip('. ')
                if self._looks_like_section_header(header_text):
                    before = para[:match.start()].strip()
                    header = header_text
                    after = para[match.end():].strip()
                    
                    if before:
                        result.append(before)
                    result.append(header)
                    if after:
                        # Recursively process the remainder
                        result.extend(self._split_inline_headers([after]))
                    split_occurred = True
                    break
            
            if not split_occurred:
                result.append(para)
        
        return result

    def _looks_like_section_header(self, text: str) -> bool:
        """Helper to validate if text looks like a genuine section header."""
        if not text or len(text) > 100:
            return False
        
        # Must start with a recognized pattern
        if not re.match(r'^(?:[IVX]+|[A-Z]|\d+)\.\s+', text):
            return False
        
        # Should be relatively short and title-cased
        words = text.split()
        if len(words) > 12:
            return False
        
        # Check for section keywords or proper title structure
        if self._contains_section_keyword(text):
            return True
        
        # Or check if it's title case (most words capitalized)
        capitalized = sum(1 for w in words if w and w[0].isupper())
        if capitalized >= len(words) * 0.6:
            return True
        
        return False

    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract logical sections from cleaned legal text"""
        if not text:
            return []

        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        sections: List[Dict[str, Any]] = []
        stack: List[Tuple[str, int]] = []
        current: Dict[str, Any] = {"title": "Preamble", "content": [], "level": 0, "path": "Preamble"}

        # Track dialogue sections
        in_dialogue = False
        dialogue_buffer: List[str] = []

        for para in paragraphs:
            # Handle dialogue sections
            if self._is_dialogue_line(para):
                if not in_dialogue:
                    if current["content"]:
                        assembled = "\n\n".join(current["content"]).strip()
                        if assembled:
                            sections.append({
                                "title": current["title"],
                                "content": assembled,
                                "level": current.get("level", 0),
                                "path": current.get("path", current["title"]),
                                "is_dialogue": False,
                            })
                        current["content"] = []
                    in_dialogue = True
                    dialogue_buffer = [para]
                else:
                    dialogue_buffer.append(para)
                continue
            elif in_dialogue:
                if dialogue_buffer:
                    dialogue_content = "\n\n".join(dialogue_buffer)
                    sections.append({
                        "title": "Court Transcript",
                        "content": dialogue_content,
                        "level": current.get("level", 0),
                        "path": current.get("path", "Court Transcript"),
                        "is_dialogue": True,
                    })
                    dialogue_buffer = []
                in_dialogue = False

            is_header, section_title, level = self._detect_section_header(para)

            if is_header and section_title:
                if current["content"]:
                    assembled = "\n\n".join(current["content"]).strip()
                    if assembled:
                        sections.append({
                            "title": current["title"],
                            "content": assembled,
                            "level": current.get("level", 0),
                            "path": current.get("path", current["title"]),
                            "is_dialogue": False,
                        })

                # Better hierarchy management
                if level > 0:
                    while stack and stack[-1][1] >= level:
                        stack.pop()
                    stack.append((section_title, level))
                    path = " / ".join([t for t, _lvl in stack])
                else:
                    stack = [(section_title, level)]
                    path = section_title
                current = {"title": section_title, "content": [], "level": level, "path": path}
                continue
            current["content"].append(para)

        # Handle any remaining dialogue
        if in_dialogue and dialogue_buffer:
            dialogue_content = "\n\n".join(dialogue_buffer)
            sections.append({
                "title": "Court Transcript",
                "content": dialogue_content,
                "level": current.get("level", 0),
                "path": current.get("path", "Court Transcript"),
                "is_dialogue": True,
            })

        # Handle remaining content
        if current["content"]:
            assembled = "\n\n".join(current["content"]).strip()
            if assembled:
                sections.append({
                    "title": current["title"],
                    "content": assembled,
                    "level": current.get("level", 0),
                    "path": current.get("path", current["title"]),
                    "is_dialogue": False,
                })

        return sections

    def _contains_section_keyword(self, heading: str) -> bool:
        """Return True if heading contains a known section keyword/phrase."""
        if not heading:
            return False

        normalized = re.sub(r"[^A-Z0-9\s]", " ", heading.upper())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        padded = f" {normalized} "
        return any(f" {keyword} " in padded for keyword in self.SECTION_KEYWORDS)

    def _detect_section_header(self, paragraph: str) -> Tuple[bool, Optional[str], int]:
        """Detect if a paragraph represents a section header with pattern matching."""
        text = paragraph.strip()
        if not text or len(text) > 140:
            return False, None, 0

        lower_text = text.lower()
        if "non-precedential decision" in lower_text:
            return False, None, 0
        if any(term in lower_text for term in self.CAPTION_KEYWORDS):
            return False, None, 0
        if " v. " in lower_text or " vs. " in lower_text:
            return False, None, 0

        # More flexible Roman numeral pattern
        roman_match = re.match(r"^([IVX]{1,7})\.\s+(.+)", text)
        if roman_match:
            return True, text.rstrip(":").strip(), 1

        # Alpha pattern (handles both "A." and "A. Title")
        alpha_match = re.match(r"^([A-Z])\.\s+(.+)", text)
        if alpha_match and len(alpha_match.group(2).split()) >= 2:  # Ensure there's actual content
            return True, text.rstrip(":").strip(), 2

        # Numeric patterns with sub-levels
        numeric_match = re.match(r"^(\d+)\.\s+(.+)", text)
        if numeric_match:
            if text.endswith("?") or len(text.split()) > 18:
                return False, None, 0
            return True, text.rstrip(":").strip(), 3

        # Sub-numeric patterns (e.g., "1.1", "2.3")
        subnumeric_match = re.match(r"^(\d+\.\d+)\.\s+(.+)", text)
        if subnumeric_match:
            return True, text.rstrip(":").strip(), 4

        # Colon-ending headers with better validation
        if text.endswith(":") and len(text.split()) <= 7:
            candidate = text.rstrip(":").strip()

            if "." in candidate:
                return False, None, 0
            if re.search(r"\b(id|ibid|tr|transcript)\b", candidate, flags=re.IGNORECASE):
                return False, None, 0
            if re.search(r"\bat\s+\d", candidate, flags=re.IGNORECASE):
                return False, None, 0
            if re.search(r"[\"'""'']", candidate):
                return False, None, 0
            if re.match(r"^(the|a|an)\b", candidate, flags=re.IGNORECASE):
                return False, None, 0
            if re.search(r"\b(as follows|the following|summarized|testimony|presented)\b", candidate, flags=re.IGNORECASE):
                return False, None, 0
            if not self._contains_section_keyword(candidate):
                return False, None, 0

            return True, candidate, 1

        # ALL CAPS headers
        condensed = re.sub(r"[^A-Z0-9\s:]", "", text).strip()
        words = condensed.split()
        if condensed and condensed == condensed.upper() and 1 <= len(words) <= 12:
            if self._contains_section_keyword(condensed):
                return True, text.rstrip(":").strip(), 1

        return False, None, 0

    def _is_noise_line(self, line: str) -> bool:
        """Check if a line matches noise patterns."""
        for pattern in self._noise_re:
            if pattern.match(line):
                return True
        return False

    def _merge_lines_into_paragraphs(self, lines: List[str]) -> List[str]:
        """Merge wrapped lines into coherent paragraphs."""
        paragraphs: List[str] = []
        current: List[str] = []

        for line in lines:
            if line == "":
                if current:
                    paragraphs.append(self._join_wrapped_lines(current))
                    current = []
                continue

            current.append(line)

        if current:
            paragraphs.append(self._join_wrapped_lines(current))

        return paragraphs

    def _join_wrapped_lines(self, lines: List[str]) -> str:
        """Join lines that were wrapped mid-word."""
        merged: List[str] = []
        for line in lines:
            if not merged:
                merged.append(line.strip())
                continue

            if merged[-1].endswith("-"):
                base = merged[-1][:-1]
                nxt = line.lstrip()
                if nxt and nxt[0].islower():
                    merged[-1] = base + nxt
                else:
                    merged[-1] = base + "-" + nxt
            else:
                merged.append(line.strip())

        paragraph = " ".join(merged)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        return paragraph

    def structure_opinion(self, raw_text: str) -> Dict[str, Any]:
        """Process raw opinion text into structured format."""
        metadata = self.extract_metadata(raw_text)
        cleaned = self.clean_text(raw_text)
        sections = self.extract_sections(cleaned)

        return {
            "structured_text": cleaned,
            "sections": sections,
            "word_count": len(cleaned.split()) if cleaned else 0,
            "section_count": len(sections),
            "metadata": metadata,
        }

    def get_summary_header(
        self, 
        case_name: str, 
        court_id: str, 
        docket_number: str, 
        date_filed: Optional[str]
    ) -> str:
        """Generate a summary header for prepending to chunks."""
        parts = [f"Case: {case_name}"] if case_name else []
        if court_id:
            parts.append(f"Court: {court_id}")
        if docket_number:
            parts.append(f"Docket: {docket_number}")
        if date_filed:
            parts.append(f"Filed: {date_filed}")

        return " | ".join(parts)

    def prepare_for_chunking(
        self,
        raw_text: str,
        case_name: str,
        court_id: str = "",
        docket_number: str = "",
        date_filed: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare opinion text for the chunking pipeline."""
        structured = self.structure_opinion(raw_text)
        
        extracted_metadata = structured.get("metadata", {})
        if not docket_number and extracted_metadata.get("docket_number"):
            docket_number = extracted_metadata["docket_number"]
        if not date_filed and extracted_metadata.get("date_filed"):
            date_filed = extracted_metadata["date_filed"]
        
        header = self.get_summary_header(case_name, court_id, docket_number, date_filed)

        return {
            "header": header,
            "full_text": structured["structured_text"],
            "sections": structured["sections"],
            "word_count": structured["word_count"],
            "section_count": structured["section_count"],
            "metadata": extracted_metadata,
        }