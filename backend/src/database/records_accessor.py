from datetime import datetime
from typing import List, Optional, Set
from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.database.opinion_table import OpinionsTable
from src.schemas.core_schema import CreateRecord


class RecordAccessor:
    """Data access layer for opinions.

    :param session: Active SQLAlchemy session
    """

    def __init__(self, session: Session):
        self.session = session

    def create(self, opinion: CreateRecord) -> OpinionsTable:
        """Persist a new opinion record.

        :param opinion: Opinion payload
        :returns: Created opinion row
        """
        db_opinion = OpinionsTable(**opinion.model_dump())
        self.session.add(db_opinion)
        self.session.commit()
        self.session.refresh(db_opinion)
        return db_opinion

    def get_by_opinion_id(self, opinion_id: int) -> Optional[OpinionsTable]:
        """Retrieve an opinion by CourtListener ID.

        :param opinion_id: CourtListener opinion ID
        :returns: Opinion row or None
        """
        stmt = select(OpinionsTable).where(OpinionsTable.opinion_id == opinion_id)
        return self.session.scalar(stmt)

    def get_by_id(self, db_id: UUID) -> Optional[OpinionsTable]:
        """Retrieve an opinion by database UUID.

        :param db_id: Database identifier
        :returns: Opinion row or None
        """
        stmt = select(OpinionsTable).where(OpinionsTable.id == db_id)
        return self.session.scalar(stmt)

    def get_all(self, limit: int = 100, offset: int = 0) -> List[OpinionsTable]:
        """List opinions ordered by filing date.

        :param limit: Max records to return
        :param offset: Records to skip
        :returns: Ordered opinion list
        """
        stmt = select(OpinionsTable).order_by(OpinionsTable.date_filed.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_count(self) -> int:
        """Count total opinions.

        :returns: Total record count
        """
        stmt = select(func.count(OpinionsTable.id))
        return self.session.scalar(stmt) or 0

    def get_existing_opinion_ids(self) -> Set[int]:
        """Get all existing opinion IDs for deduplication.

        :returns: Set of CourtListener opinion IDs
        """
        stmt = select(OpinionsTable.opinion_id)
        return set(self.session.scalars(stmt))

    def get_opinions_with_text(self, limit: int = 100, offset: int = 0) -> List[OpinionsTable]:
        """Get opinions that have structured text.

        :param limit: Max records to return
        :param offset: Records to skip
        :returns: Opinions with structured text
        """
        stmt = (
            select(OpinionsTable)
            .where(OpinionsTable.structured_text.is_not(None))
            .order_by(OpinionsTable.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt))

    def get_recent_opinions(self, limit: int = 100) -> List[OpinionsTable]:
        """Get recently added opinions.

        :param limit: Max records to return
        :returns: Recent opinions ordered by creation date
        """
        stmt = select(OpinionsTable).order_by(OpinionsTable.created_at.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def get_stats(self) -> dict:
        """Summarize opinion statistics.

        :returns: Aggregated stats
        """
        total = self.get_count()
        text_stmt = select(func.count(OpinionsTable.id)).where(OpinionsTable.structured_text.is_not(None))
        with_text = self.session.scalar(text_stmt) or 0

        return {
            "total_opinions": total,
            "opinions_with_structured_text": with_text,
            "processing_rate": (with_text / total * 100) if total > 0 else 0,
        }

    def update(self, opinion: OpinionsTable) -> OpinionsTable:
        """Persist updates to an opinion row.

        :param opinion: Opinion entity to save
        :returns: Updated opinion row
        """
        self.session.add(opinion)
        self.session.commit()
        self.session.refresh(opinion)
        return opinion

    def upsert(self, opinion_create: CreateRecord) -> OpinionsTable:
        """Insert or update an opinion record.

        :param opinion_create: Opinion payload
        :returns: Saved opinion row
        """
        existing = self.get_by_opinion_id(opinion_create.opinion_id)
        if existing:
            for key, value in opinion_create.model_dump(exclude_unset=True).items():
                setattr(existing, key, value)
            return self.update(existing)
        return self.create(opinion_create)

