from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PostgreSQLSettings(BaseSettings):
    database_url: str = Field(..., description="PostgreSQL database URL")
    echo_sql: bool = Field(..., description="Enable SQL query logging")
    pool_size: int = Field(..., description="Database connection pool size")
    max_overflow: int = Field(..., description="Maximum pool overflow")

    class Config:
        env_prefix = "POSTGRES_"


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    chunk_index: int = Field(..., description="Zero-based index of the chunk")
    start_char: int = Field(..., description="Start character offset")
    end_char: int = Field(..., description="End character offset")
    word_count: int = Field(..., description="Number of words in the chunk")
    overlap_with_previous: int = Field(..., description="Overlap with previous chunk")
    overlap_with_next: int = Field(..., description="Overlap with next chunk")
    section_title: Optional[str] = Field(None, description="Section title if available")
    section_path: Optional[str] = None   
    section_level: int = 0
    parent_section: Optional[str] = None
    is_dialogue: bool = False   


class TextChunk(BaseModel):
    """A chunk of text from an opinion."""
    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    opinion_id: int = Field(..., description="CourtListener opinion ID")
    case_id: str = Field(..., description="Internal database ID")
