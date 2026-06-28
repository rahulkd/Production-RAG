# Drop all tables in the PostgreSQL `rag_db` database.
from sqlalchemy import inspect

import src.models.paper  # noqa: F401 - register models on Base.metadata
from src.db.factory import make_database
from src.db.interfaces.postgresql import Base


def delete_tables() -> None:
    """Drop every table defined on the SQLAlchemy metadata from `rag_db`.

    Lists the tables that currently exist, asks for confirmation, then drops
    them. This is destructive and removes all stored papers.
    """
    database = make_database()

    assert database.engine is not None
    inspector = inspect(database.engine)
    existing_tables = inspector.get_table_names()

    if not existing_tables:
        print("No tables found in the database - nothing to delete.")
        database.teardown()
        return

    print(f"Tables to be dropped: {', '.join(existing_tables)}")
    confirmation = input("Drop these tables? This cannot be undone. [y/N]: ").strip().lower()

    if confirmation != "y":
        print("Aborted - no tables were dropped.")
        database.teardown()
        return

    Base.metadata.drop_all(bind=database.engine)

    remaining_tables = inspect(database.engine).get_table_names()
    dropped = set(existing_tables) - set(remaining_tables)

    print(f"Dropped {len(dropped)} table(s): {', '.join(sorted(dropped)) if dropped else 'None'}")
    if remaining_tables:
        print(f"Remaining tables: {', '.join(remaining_tables)}")

    database.teardown()


if __name__ == "__main__":
    delete_tables()
