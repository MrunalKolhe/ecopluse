import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# ── Database URL ──────────────────────────────────────────────────────────────
# On Railway, DATABASE_URL is automatically injected as an env variable.
# Locally, set it in backend/.env (e.g. DATABASE_URL=postgresql://user:pass@host/db)
# Railway sometimes provides postgres:// URLs — SQLAlchemy requires postgresql://
_raw_url = os.getenv("DATABASE_URL", "")
if not _raw_url:
    raise RuntimeError(
        "❌ DATABASE_URL environment variable is not set. "
        "Add it in Railway → Variables, or in backend/.env for local dev."
    )

# Fix legacy 'postgres://' scheme — SQLAlchemy 1.4+ requires 'postgresql://'
SQLALCHEMY_DATABASE_URL = _raw_url.replace("postgres://", "postgresql://", 1)
logger.info(f"✅ Database URL loaded (driver: {SQLALCHEMY_DATABASE_URL.split('://')[0]})")

# ── Engine ────────────────────────────────────────────────────────────────────
# No connect_args needed for PostgreSQL (check_same_thread is SQLite-only)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,   # reconnect automatically if connection drops
    pool_size=5,
    max_overflow=10,
)

# ── Session & Base ────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and ensures it is closed."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
