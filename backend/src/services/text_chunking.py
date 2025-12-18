import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from src.schemas.opinion_schema import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)

@dataclass
class _Sentence:
    text: str
    word_count: int
    start_char: int
    end_char: int
    paragraph_start: bool

class TextChunker:
    """Semantics-aware chunker for processed cleaned text"""

    ABBREVIATIONS = {"u.s.","u.s.c","v.","vs.","no.","nos.","mr.","ms.","mrs.","dr.","prof.","inc.","llc.","l.l.c.","co.","corp.",
        "dept.","dep't.","ass'n.","fig.","al.","jr.","sr.","id.","ibid.","cf.","e.g.","i.e.","etc.","seq."}

    LEGAL_CITATION_PATTERNS = [r'\bId\.', r'\bSee\.', r'\bcf\.', r'\bCf\.', r'\be\.g\.',r'\bi\.e\.', r'\betc\.', r'\bseq\.', r'\bIbid\.']
    SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z0-9\"'(\[])")
    METADATA_SECTION_PATTERNS = [r"^opinion\s*header$", r"^caption$", r"^syllabus$", r"^parties$", r"^counsel$", r"^before:", r"^appearing\s+for"]
    SUBSTANTIVE_PREAMBLE_PATTERNS = [r"procedural\s+history", r"background", r"facts", r"introduction"]

    def __init__(
        self, 
        chunk_size: int = 600, 
        overlap_size: int = 150, 
        min_chunk_size: int = 350 
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.sentence_overlap = max(1, min(4, overlap_size // 40 or 1))

        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")

    def _section_title_for_display(self, section_title: Optional[str]) -> Optional[str]:
        """Return a retrieval-friendly section title for inclusion in chunk text."""
        if not section_title:
            return None

        title = " ".join(str(section_title).split()).strip()
        if not title:
            return None

        if len(title) > 300:
            return None

        if re.search(r"\b(id|ibid)\.(?=\s|$)", title, flags=re.IGNORECASE):
            return None
        if re.search(r"\btr\.(?=\s|$)|\btranscript\b", title, flags=re.IGNORECASE):
            return None
        if re.search(r"\bat\s+\d", title, flags=re.IGNORECASE):
            return None

        return title

    def chunk_opinion(
        self,
        case_name: str,
        full_text: str,
        opinion_id: int,
        case_id: str,
        court_id: str = "",
        docket_number: str = "",
        date_filed: Optional[str] = None,
        sections: Optional[Union[Dict[str, str], str, list]] = None,
    ) -> List[TextChunk]:
        """Chunk a legal opinion, preferring section-aware chunking when provided."""
        if sections:
            try:
                section_chunks = self._chunk_by_sections(
                    case_name=case_name,
                    court_id=court_id,
                    docket_number=docket_number,
                    date_filed=date_filed,
                    opinion_id=opinion_id,
                    case_id=case_id,
                    sections=sections,
                )
                if section_chunks:
                    return section_chunks
            except Exception:
                logger.exception("Failed to chunk by sections; falling back to full text chunking")

        return self.chunk_text(
            text=full_text,
            opinion_id=opinion_id,
            case_id=case_id,
            case_name=case_name,
            court_id=court_id,
            docket_number=docket_number,
            date_filed=date_filed,
        )

    def chunk_text(
        self,
        text: str,
        opinion_id: int,
        case_id: str,
        case_name: str = "",
        court_id: str = "",
        docket_number: str = "",
        date_filed: Optional[str] = None,
        section_title: Optional[str] = None,
        section_path: Optional[str] = None, 
        section_level: int = 0, 
        parent_section: Optional[str] = None,
        is_dialogue: bool = False,
        chunk_index_offset: int = 0,
        header_override: Optional[str] = None,
    ) -> List[TextChunk]:
        """Chunk text into overlapping segments while respecting sentences and paragraphs."""
        sanitized = self._sanitize_text(text)
        if not sanitized:
            return []

        header = header_override if header_override is not None else self._build_header(case_name, court_id, docket_number, date_filed)
        sentences = self._prepare_sentences(sanitized)
        if is_dialogue:
            groups = self._group_dialogue_sentences(sentences)
        else:
            groups = self._group_sentences(sentences)
        if len(groups) >= 2:
            last_chunk = groups[-1]
            second_last = groups[-2]
            
            # More aggressive merging for very small trailing chunks
            if last_chunk["word_count"] < self.min_chunk_size:
                second_last["sentences"].extend(last_chunk["sentences"])
                second_last["overlap_next_words"] = 0
                second_last["word_count"] = sum(s.word_count for s in second_last["sentences"])
                groups.pop()

        return self._render_chunks(
            groups=groups,
            header=header,
            opinion_id=opinion_id,
            case_id=case_id,
            section_title=section_title,
            section_path=section_path,
            section_level=section_level,
            parent_section=parent_section,
            is_dialogue=is_dialogue,
            chunk_index_offset=chunk_index_offset,
        )

    def _sanitize_text(self, text: str) -> str:
        """Sanitize and normalize text."""
        sanitized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized

    def _prepare_sentences(self, text: str) -> List[_Sentence]:
        """Parse text into sentence objects."""
        paragraphs = self._split_paragraphs(text)
        sentences: List[_Sentence] = []

        for para_idx, (para_text, para_start, _) in enumerate(paragraphs):
            para_sentences = self._split_sentences_in_paragraph(para_text, para_start)
            for idx, sent in enumerate(para_sentences):
                sentences.append(
                    _Sentence(
                        text=sent["text"],
                        word_count=len(sent["text"].split()),
                        start_char=sent["start"],
                        end_char=sent["end"],
                        paragraph_start=idx == 0 and para_idx > 0,
                    )
                )

        return [s for s in sentences if s.text]

    def _split_paragraphs(self, text: str) -> List[tuple]:
        """Split text into paragraphs with position tracking."""
        parts = text.split("\n\n")
        paragraphs = []
        cursor = 0

        for idx, raw in enumerate(parts):
            para = raw.strip()
            if not para:
                cursor += len(raw) + 2
                continue

            start = cursor
            end = start + len(para)
            paragraphs.append((para, start, end))
            cursor = end + 2

        return paragraphs

    def _is_legal_citation_context(self, text: str, position: int) -> bool:
        """Check if a period at the given position is part of a legal citation."""
        window_start = max(0, position - 20)
        window_end = min(len(text), position + 5)
        window = text[window_start:window_end]
        
        for pattern in self.LEGAL_CITATION_PATTERNS:
            if re.search(pattern, window):
                return True
        
        if re.search(r'\.\s+at\s+\d', window):
            return True
        
        return False

    def _split_sentences_in_paragraph(self, paragraph: str, base_start: int) -> List[Dict[str, int]]:
        """Split paragraph into sentences, handling legal citations and numbered lists."""
        numbered_list_pattern = re.compile(r'^\s*\(\d+\)\s+')
        if numbered_list_pattern.match(paragraph):
            return [{"text": paragraph.strip(), "start": base_start, "end": base_start + len(paragraph)}]

        parts = []
        start = 0
        
        for match in self.SENTENCE_BOUNDARY_PATTERN.finditer(paragraph):
            end = match.start()
            
            if self._is_legal_citation_context(paragraph, end):
                continue
            
            if end > 0 and paragraph[end - 1].isdigit():
                continue
            
            sentence_text = paragraph[start:end].strip()
            if sentence_text:
                parts.append({"text": sentence_text, "start": base_start + start, "end": base_start + end})
            start = match.end()

        tail = paragraph[start:].strip()
        if tail:
            parts.append({"text": tail, "start": base_start + start, "end": base_start + len(paragraph)})

        merged: List[Dict[str, int]] = []
        i = 0
        while i < len(parts):
            current = parts[i]
            if self._ends_with_abbreviation(current["text"]) and i + 1 < len(parts):
                nxt = parts[i + 1]
                merged_text = f"{current['text']} {nxt['text']}"
                merged.append({
                    "text": merged_text,
                    "start": current["start"],
                    "end": nxt["end"],
                })
                i += 2
            else:
                merged.append(current)
                i += 1

        return merged

    def _ends_with_abbreviation(self, sentence: str) -> bool:
        """Check if sentence ends with a legal abbreviation."""
        if not sentence:
            return False
        tokens = sentence.rstrip().split()
        if not tokens:
            return False
        last = tokens[-1].lower()
        last = last.rstrip(").;,")
        return last in self.ABBREVIATIONS

    def _group_sentences(self, sentences: List[_Sentence]) -> List[Dict]:
        """Group sentences into chunks with overlap."""
        groups: List[Dict] = []
        current: List[_Sentence] = []
        current_words = 0

        for sentence in sentences:
            words = sentence.word_count
            projected = current_words + words

            if current and projected > self.chunk_size and current_words >= self.min_chunk_size:
                overlap = self._select_overlap_validated(current)
                overlap_words = sum(s.word_count for s in overlap)
                groups.append({
                    "sentences": current,
                    "overlap_next_words": overlap_words,
                    "word_count": current_words,
                })
                current = overlap.copy()
                current_words = overlap_words

            current.append(sentence)
            current_words += words

        if current:
            groups.append({
                "sentences": current,
                "overlap_next_words": 0,
                "word_count": current_words,
            })

        return groups

    def _group_dialogue_sentences(self, sentences: List[_Sentence]) -> List[Dict]:
        """Group dialogue/transcript sentences, keeping speaker exchanges together."""
        groups: List[Dict] = []
        current: List[_Sentence] = []
        current_words = 0
        
        # Pattern to detect speaker changes
        speaker_pattern = re.compile(r'^(MR\.|MS\.|MRS\.|THE\s+COURT|JUDGE)', re.IGNORECASE)

        for sentence in sentences:
            words = sentence.word_count
            projected = current_words + words
            
            # Check if this is a new speaker
            is_new_speaker = speaker_pattern.match(sentence.text)
            
            # Split chunks at speaker boundaries when near chunk size
            if current and projected > self.chunk_size and is_new_speaker:
                overlap = self._select_overlap_validated(current)
                overlap_words = sum(s.word_count for s in overlap)
                groups.append({
                    "sentences": current,
                    "overlap_next_words": overlap_words,
                    "word_count": current_words,
                })
                current = overlap.copy()
                current_words = overlap_words

            current.append(sentence)
            current_words += words

        if current:
            groups.append({
                "sentences": current,
                "overlap_next_words": 0,
                "word_count": current_words,
            })

        return groups

    def _select_overlap_validated(self, sentences: List[_Sentence]) -> List[_Sentence]:
        """Select overlap sentences with validation to match target overlap size."""
        if not sentences:
            return []
        
        overlap_sentences = sentences[-self.sentence_overlap:]
        overlap_words = sum(s.word_count for s in overlap_sentences)
        min_target = int(self.overlap_size * 0.7)
        max_target = int(self.overlap_size * 1.3)
        
        if overlap_words < min_target and self.sentence_overlap < len(sentences):
            test_overlap = sentences[-(self.sentence_overlap + 1):]
            test_words = sum(s.word_count for s in test_overlap)
            if test_words <= max_target:
                return test_overlap
        
        elif overlap_words > max_target and self.sentence_overlap > 1:
            test_overlap = sentences[-(self.sentence_overlap - 1):]
            test_words = sum(s.word_count for s in test_overlap)
            if test_words >= min_target:
                return test_overlap
        
        return overlap_sentences

    def _render_chunks(
        self,
        groups: List[Dict],
        header: str,
        opinion_id: int,
        case_id: str,
        section_title: Optional[str],
        section_path: Optional[str],
        section_level: int,
        parent_section: Optional[str],
        is_dialogue: bool,
        chunk_index_offset: int,
    ) -> List[TextChunk]:
        """Render sentence groups into TextChunk objects."""
        chunks: List[TextChunk] = []

        for idx, group in enumerate(groups):
            overlap_prev = groups[idx - 1]["overlap_next_words"] if idx > 0 else 0
            overlap_next = group["overlap_next_words"] if idx < len(groups) - 1 else 0
            chunk = self._build_chunk(
                sentences=group["sentences"],
                header=header,
                opinion_id=opinion_id,
                case_id=case_id,
                section_title=section_title,
                section_path=section_path,
                section_level=section_level,
                parent_section=parent_section,
                is_dialogue=is_dialogue,
                chunk_index=chunk_index_offset + idx,
                overlap_prev=overlap_prev,
                overlap_next=overlap_next,
            )
            chunks.append(chunk)

        return chunks

    def _build_chunk(
        self,
        sentences: List[_Sentence],
        header: str,
        opinion_id: int,
        case_id: str,
        section_title: Optional[str],
        section_path: Optional[str],
        section_level: int,
        parent_section: Optional[str],
        is_dialogue: bool,
        chunk_index: int,
        overlap_prev: int,
        overlap_next: int,
    ) -> TextChunk:
        """Build a single TextChunk with enhanced metadata."""
        body_parts: List[str] = []
        for i, sentence in enumerate(sentences):
            if i > 0:
                if sentence.paragraph_start:
                    body_parts.append("\n\n")
                else:
                    body_parts.append(" ")
            body_parts.append(sentence.text)

        body_text = "".join(body_parts).strip()
        display_section_title = self._section_title_for_display(section_title)
        
        # Use full path if available, otherwise fall back to title
        display_label = section_path if section_path else display_section_title
        if display_label:
            body_text = f"Section: {display_label}\n\n{body_text}"

        chunk_text = f"{header}\n\n{body_text}" if header else body_text
        start_char = sentences[0].start_char if sentences else 0
        end_char = sentences[-1].end_char if sentences else 0
        word_count = len(body_text.split())

        return TextChunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                word_count=word_count,
                overlap_with_previous=overlap_prev,
                overlap_with_next=overlap_next,
                section_title=section_title,
                section_path=section_path,  
                section_level=section_level,
                parent_section=parent_section, 
                is_dialogue=is_dialogue,
            ),
            opinion_id=opinion_id,
            case_id=case_id,
        )

    def _build_header(
        self, 
        case_name: str, 
        court_id: str, 
        docket_number: str,
        date_filed: Optional[str] = None
    ) -> str:
        """Build chunk header with all available metadata."""
        parts = []
        if case_name:
            parts.append(f"Case: {case_name}")
        if court_id:
            parts.append(f"Court: {court_id}")
        if docket_number:
            parts.append(f"Docket: {docket_number}")
        if date_filed:
            parts.append(f"Filed: {date_filed}")

        return " | ".join(parts) if parts else ""

    def _is_metadata_section(self, section_title: str, section_content: str = "") -> bool:
        """Determine if a section is metadata/non-substantive content."""
        if not section_title:
            return False
        
        title_lower = section_title.lower().strip()
        
        # Check for substantive preambles first (override metadata classification)
        for pattern in self.SUBSTANTIVE_PREAMBLE_PATTERNS:
            if re.search(pattern, title_lower):
                return False
        
        # Check standard metadata patterns
        for pattern in self.METADATA_SECTION_PATTERNS:
            if re.match(pattern, title_lower):
                return True
        
        # Check content length and density
        # True metadata sections are usually very short and contain mostly names/dates
        if section_content:
            words = section_content.split()
            if len(words) < 50:
                capitalized = sum(1 for w in words if w and w[0].isupper())
                if capitalized / len(words) > 0.7:
                    return True
        
        return False

    def _chunk_by_sections(
        self,
        case_name: str,
        court_id: str,
        docket_number: str,
        date_filed: Optional[str],
        opinion_id: int,
        case_id: str,
        sections: Union[Dict[str, str], str, list],
    ) -> List[TextChunk]:
        """Chunk by sections with improved hierarchy handling and merging logic."""
        sections_list = self._parse_sections(sections)
        if not sections_list:
            return []

        header = self._build_header(case_name, court_id, docket_number, date_filed)
        chunks: List[TextChunk] = []
        pending_sections: List[str] = []
        pending_titles: List[str] = []
        pending_paths: List[str] = []
        pending_is_dialogue = False

        def _flush_pending() -> None:
            nonlocal pending_sections, pending_titles, pending_paths, pending_is_dialogue, chunks
            if not pending_sections:
                return
            combined_content = "\n\n".join(pending_sections).strip()
            combined_title = pending_titles[0] if len(pending_titles) == 1 else " + ".join(pending_titles)
            combined_path = pending_paths[0] if len(pending_paths) == 1 else " + ".join(pending_paths)
            
            chunks.extend(
                self.chunk_text(
                    text=combined_content,
                    opinion_id=opinion_id,
                    case_id=case_id,
                    section_title=combined_title,
                    section_path=combined_path,
                    section_level=sections_list[0].get("level", 0) if pending_titles else 0,
                    parent_section=self._extract_parent_section(combined_path),
                    is_dialogue=pending_is_dialogue,
                    chunk_index_offset=len(chunks),
                    header_override=header,
                )
            )
            pending_sections = []
            pending_titles = []
            pending_paths = []
            pending_is_dialogue = False

        for i, section in enumerate(sections_list):
            leaf_title = section.get("title", f"Section {i + 1}")
            display_path = section.get("path") or leaf_title
            section_content = str(section.get("content", "")).strip()
            is_dialogue = section.get("is_dialogue", False)
            
            if not section_content:
                continue

            section_words = len(section_content.split())
            is_metadata = self._is_metadata_section(leaf_title, section_content)
            
            # Metadata sections: merge with pending
            if is_metadata:
                pending_sections.append(section_content)
                pending_titles.append(leaf_title)
                pending_paths.append(display_path)
                continue
            
            # Dialogue sections: always keep separate (don't merge)
            if is_dialogue:
                _flush_pending()
                chunks.extend(
                    self.chunk_text(
                        text=section_content,
                        opinion_id=opinion_id,
                        case_id=case_id,
                        section_title=leaf_title,
                        section_path=display_path,
                        section_level=section.get("level", 0),
                        parent_section=self._extract_parent_section(display_path),
                        is_dialogue=True,
                        chunk_index_offset=len(chunks),
                        header_override=header,
                    )
                )
                continue
            
            if section_words < 100:
                should_merge = True
                # Don't merge if this is a sibling section at the same level
                if pending_paths:
                    current_level = section.get("level", 0)
                    # Parse level from last pending path
                    last_pending_level = self._get_section_level(pending_paths[-1])
                    if current_level == last_pending_level and current_level > 1:
                        # Sibling subsections - flush and start new
                        should_merge = False
                
                if should_merge:
                    pending_sections.append(section_content)
                    pending_titles.append(leaf_title)
                    pending_paths.append(display_path)
                    continue
                else:
                    _flush_pending()

            # Large substantive section: flush pending first, then chunk
            _flush_pending()
            chunks.extend(
                self.chunk_text(
                    text=section_content,
                    opinion_id=opinion_id,
                    case_id=case_id,
                    section_title=leaf_title,
                    section_path=display_path,
                    section_level=section.get("level", 0),
                    parent_section=self._extract_parent_section(display_path),
                    is_dialogue=False,
                    chunk_index_offset=len(chunks),
                    header_override=header,
                )
            )

        _flush_pending()
        return chunks

    def _extract_parent_section(self, section_path: Optional[str]) -> Optional[str]:
        """Extract the immediate parent section from a hierarchical path."""
        if not section_path or "/" not in section_path:
            return None
        
        parts = section_path.split(" / ")
        if len(parts) <= 1:
            return None
        
        return parts[-2].strip()

    def _get_section_level(self, section_path: str) -> int:
        """Estimate section level from path depth."""
        if not section_path:
            return 0
        return section_path.count(" / ") + 1

    def _parse_sections(self, sections: Union[Dict[str, str], str, list]) -> List[Dict[str, Any]]:
        """Parse sections from various input formats."""
        if isinstance(sections, list):
            result: List[Dict[str, Any]] = []
            for i, section in enumerate(sections):
                if isinstance(section, dict):
                    title = section.get("title", section.get("heading", f"Section {i + 1}"))
                    content = section.get("content", section.get("text", ""))
                    parsed: Dict[str, Any] = {"title": title, "content": content}
                    if "path" in section:
                        parsed["path"] = section.get("path")
                    if "level" in section:
                        parsed["level"] = section.get("level")
                    if "is_dialogue" in section:
                        parsed["is_dialogue"] = section.get("is_dialogue")
                    result.append(parsed)
                else:
                    result.append({"title": f"Section {i + 1}", "content": str(section)})
            return result
        if isinstance(sections, dict):
            return [{"title": k, "content": v} for k, v in sections.items()]
        if isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                return self._parse_sections(parsed)
            except json.JSONDecodeError:
                return []
        return []