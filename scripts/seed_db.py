"""Generate the synthetic corpus and load structured tables (idempotent).

Usage:
    python scripts/seed_db.py                 # corpus + DB, truncate-and-reload
    python scripts/seed_db.py --corpus-only   # only write data/seed/
    python scripts/seed_db.py --db-only       # only load the database
    python scripts/seed_db.py --no-truncate   # append instead of reset (rarely wanted)
    python scripts/seed_db.py --seed 42       # override the data seed
"""

from __future__ import annotations

import argparse
import asyncio

from app.core.db import dispose_engine
from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings
from app.seed.generator import DEFAULT_SEED_DIR, generate_dataset, write_corpus
from app.seed.loader import load_database

logger = get_logger("seed")


async def _run(seed: int, *, corpus: bool, database: bool, truncate: bool) -> None:
    dataset = generate_dataset(seed)
    logger.info(
        "seed.generated",
        extra={
            "seed": seed,
            "incidents": len(dataset.incidents),
            "tickets": len(dataset.tickets),
            "postmortems": len(dataset.postmortems),
            "runbooks": len(dataset.runbooks),
            "metrics_rows": len(dataset.metrics),
        },
    )

    if corpus:
        files = write_corpus(dataset, DEFAULT_SEED_DIR)
        logger.info("seed.corpus_written", extra={"files": files, "dir": str(DEFAULT_SEED_DIR)})

    if database:
        try:
            counts = await load_database(dataset, truncate=truncate)
            logger.info("seed.database_loaded", extra={"counts": counts, "truncate": truncate})
        finally:
            await dispose_engine()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Seed synthetic ops data.")
    parser.add_argument("--seed", type=int, default=settings.data_seed)
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--db-only", action="store_true")
    parser.add_argument("--no-truncate", action="store_true")
    args = parser.parse_args()

    corpus = not args.db_only
    database = not args.corpus_only
    asyncio.run(
        _run(
            args.seed,
            corpus=corpus,
            database=database,
            truncate=not args.no_truncate,
        )
    )


if __name__ == "__main__":
    main()
