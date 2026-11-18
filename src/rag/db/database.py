import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool, StaticPool

from src.config import get_logger
from src.rag.config import DATABASE_URL

logger = get_logger(__name__)

Base = declarative_base()

_ENGINE = None
_SESSION_FACTORY = None


def get_engine(db_url: Optional[str] = None):
    """
    Returns a configured SQLAlchemy engine singleton.
    Supports PostgreSQL with connection pooling and SQLite for test isolation.
    """
    global _ENGINE, _SESSION_FACTORY
    target_url = db_url or DATABASE_URL

    if _ENGINE is None or (db_url and str(_ENGINE.url) != db_url):
        if target_url.startswith("sqlite"):
            _ENGINE = create_engine(
                target_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            _ENGINE = create_engine(
                target_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_pre_ping=True
            )
        _SESSION_FACTORY = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)
        )
        logger.info(f"Initialized Database engine for: {_ENGINE.url.render_as_string(hide_password=True)}")

    return _ENGINE


def get_session(db_url: Optional[str] = None) -> Session:
    """Returns a new database session."""
    get_engine(db_url)
    return _SESSION_FACTORY()


@contextmanager
def get_db(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Context manager providing a transactional database session with
    automatic commit on success and rollback on exceptions.
    """
    session = get_session(db_url)
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        session.close()


def is_db_connected(db_url: Optional[str] = None) -> bool:
    """Verifies live database connectivity."""
    try:
        engine = get_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        return True
    except Exception as e:
        logger.warning(f"Database connectivity check failed: {e}")
        return False


def init_db(db_url: Optional[str] = None, drop_all: bool = False):
    """Initializes all database tables from metadata."""
    from src.rag.db import models  # Ensure all models are registered on Base
    engine = get_engine(db_url)
    if drop_all:
        logger.warning("Dropping all existing knowledge base tables...")
        Base.metadata.drop_all(bind=engine)
    logger.info("Creating knowledge base tables if not present...")
    Base.metadata.create_all(bind=engine)
    logger.info("Knowledge base tables initialized successfully.")
