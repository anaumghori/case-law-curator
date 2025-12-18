import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from src.database.postgresql import Base


class CaseLawOpinion(Base):
    """Opinions record with metadata"""
    __tablename__ = "case_law_opinions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    opinion_id = Column(Integer, unique=True, nullable=False, index=True)
    cluster_id = Column(Integer, nullable=False, index=True)
    case_name = Column(String, nullable=False)
    plain_text = Column(Text, nullable=False)
    opinion_type = Column(String, nullable=False)
    citations = Column(JSON, nullable=False, default=list)
    date_filed = Column(DateTime, nullable=True)
    date_filed_is_approximate = Column(Boolean, default=False, nullable=False)
    precedential_status = Column(String, nullable=False)
    citation_count = Column(Integer, default=0, nullable=False)
    court_id = Column(String, nullable=False)
    docket_number = Column(String, nullable=False)
    nature_of_suit = Column(String, nullable=True, default="")
    structured_text = Column(Text, nullable=True)
    sections = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

