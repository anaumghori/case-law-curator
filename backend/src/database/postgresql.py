import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from src.config import Settings, get_settings
from src.database.database_core import BaseDatabase
from src.schemas.core_schema import PostgreSQLSettings

logger = logging.getLogger(__name__)

Base = declarative_base()


class PostgreSQLDatabase(BaseDatabase):
    """PostgreSQL-backed database adapter.

    :param config: Connection configuration
    """

    def __init__(self, config: PostgreSQLSettings):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None

    def startup(self) -> None:
        """Initialize the database connection.

        :returns: None
        """
        try:
            self.engine = create_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
            )
            self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            inspector = inspect(self.engine)
            existing_tables = set(inspector.get_table_names())
            Base.metadata.create_all(bind=self.engine)
            created_tables = set(inspector.get_table_names()) - existing_tables

            if created_tables:
                logger.info("Created database tables: %s", ", ".join(sorted(created_tables)))
        except Exception as exc:
            logger.error("Failed to initialize PostgreSQL: %s", exc)
            raise

    def teardown(self) -> None:
        """Dispose pooled connections.

        :returns: None
        """
        if self.engine:
            self.engine.dispose()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a session context manager.

        :returns: SQLAlchemy session
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call startup() first.")

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def make_database(settings: Optional[Settings] = None) -> PostgreSQLDatabase:
    """Create and start a PostgreSQL database instance.

    :param settings: Optional application settings override
    :returns: Initialized database adapter
    """
    active_settings = settings or get_settings()
    config = PostgreSQLSettings(
        database_url=active_settings.postgres_database_url,
        echo_sql=active_settings.postgres_echo_sql,
        pool_size=active_settings.postgres_pool_size,
        max_overflow=active_settings.postgres_max_overflow,
    )

    database = PostgreSQLDatabase(config=config)
    database.startup()
    return database
