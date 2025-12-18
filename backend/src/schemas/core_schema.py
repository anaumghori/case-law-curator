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


class CreateRecord(BaseModel):
    """Schema for creating an opinion record in the database."""
    opinion_id: int = Field(..., description="CourtListener opinion ID")
    cluster_id: int = Field(..., description="CourtListener cluster ID")
    case_name: str = Field(..., description="Name of the case")
    plain_text: str = Field(..., description="Plain text content")
    opinion_type: str = Field(..., description="Type of opinion")
    citations: List[str] = Field(default_factory=list, description="Case citations")
    date_filed: Optional[datetime] = Field(None, description="Filing date")
    date_filed_is_approximate: bool = Field(False, description="Whether date is approximate")
    precedential_status: str = Field(..., description="Precedential status")
    citation_count: int = Field(0, description="Number of citations")
    court_id: str = Field(..., description="Court identifier")
    docket_number: str = Field(..., description="Docket number")
    nature_of_suit: str = Field("", description="Nature of suit")
    structured_text: Optional[str] = Field(None, description="Cleaned structured text")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="Parsed sections")


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
