from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, ContextManager, Dict, Generator, List, Optional
from sqlalchemy.orm import Session

_database_instance: Optional["BaseDatabase"] = None


class BaseDatabase(ABC):
    """Core contract for database adapters.

    :returns: None
    """

    @abstractmethod
    def startup(self) -> None:
        """Initialize the database connection.

        :returns: None
        """

    @abstractmethod
    def teardown(self) -> None:
        """Close the database connection.

        :returns: None
        """

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        """Provide a database session context manager.

        :returns: SQLAlchemy session context
        """


class BaseRepository(ABC):
    """Lightweight repository helper for CRUD patterns.

    :param session: Active SQLAlchemy session
    """

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record.

        :param data: Payload to persist
        :returns: Created entity
        """

    @abstractmethod
    def get_by_id(self, record_id: Any) -> Optional[Any]:
        """Retrieve a record by identifier.

        :param record_id: Primary key value
        :returns: Retrieved entity or None
        """

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        """Update a record by identifier.

        :param record_id: Primary key value
        :param data: Fields to update
        :returns: Updated entity or None
        """

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record by identifier.

        :param record_id: Primary key value
        :returns: True when deletion succeeds
        """

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List records with pagination.

        :param limit: Maximum records to return
        :param offset: Records to skip
        :returns: List of entities
        """


def get_database() -> "BaseDatabase":
    """Get or create the shared database instance.

    :returns: Active database adapter
    """
    global _database_instance
    if _database_instance is None:
        from src.database.postgresql import make_database

        _database_instance = make_database()
    return _database_instance


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session context.

    :returns: Database session
    """
    database = get_database()
    with database.get_session() as session:
        yield session

