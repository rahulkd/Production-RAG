import logging
from contextlib import contextmanager
from typing import Generator, Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from src.db.interfaces.base import BaseDatabase

## why sqlalchemy?
# SQLAlchemy is a powerful and flexible Object-Relational Mapping (ORM) library for Python. It provides a high-level interface for working with databases, allowing developers to 
#  interact with databases using Python objects and classes instead of writing raw SQL queries. Here are some reasons why SQLAlchemy is a popular choice for database 
# interactions in Python applications:  
# 1. **Abstraction**: SQLAlchemy abstracts away the complexities of database interactions, allowing developers to work with Python objects and classes instead of writing raw SQL 
# queries. This can lead to cleaner and more maintainable code.
# 2. **Database Agnosticism**: SQLAlchemy supports multiple database backends (e.g., PostgreSQL, MySQL, SQLite), making it easier to switch between databases without changing 
# the application code.
# 3. **ORM Capabilities**: SQLAlchemy's ORM allows developers to define database schemas as Python classes, making it easier to manage database models and relationships. 
# It also provides powerful querying capabilities through its query API.
# 4. **Connection Pooling**: SQLAlchemy provides built-in support for connection pooling, which can improve performance by reusing database connections instead of creating new ones 
# for each request.
# 5. **Migrations**: SQLAlchemy can be used in conjunction with migration tools like Alembic to manage database schema changes over time, making it easier to evolve the database 
# schema as the application grows.
# 6. **Community and Ecosystem**: SQLAlchemy has a large and active community, which means there are many resources, tutorials, and third-party libraries available to help developers 
# get started and solve common problems when working with databases in Python.

logger = logging.getLogger(__name__)

## why pydantic settings?
# Pydantic Settings is a library that provides a structured way to manage application configuration using Pydantic models. 
# It allows developers to define configuration settings as Python classes, which can be easily validated
class PostgreSQLSettings(BaseSettings):
    """PostgreSQL configuration settings."""

    database_url: str = Field(
        default="postgresql://rag_user:rag_password@localhost:5432/rag_db", description="PostgreSQL database URL"
    )
    echo_sql: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=0, description="Maximum pool overflow")

    class Config:
        env_prefix = "POSTGRES_"


Base = declarative_base()


class PostgreSQLDatabase(BaseDatabase):
    """PostgreSQL database implementation."""

    def __init__(self, config: PostgreSQLSettings):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None

    def startup(self) -> None:
        """Initialize the database connection."""
        try:
            # Log connection attempt
            logger.info(
                f"Attempting to connect to PostgreSQL at: {self.config.database_url.split('@')[1] if '@' in self.config.database_url else 'localhost'}"
            )

            self.engine = create_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,  # Verify connections before use
            )

            ## what is sessionmaker?
            # `sessionmaker` is a factory function provided by SQLAlchemy that creates a new session class. A session in SQLAlchemy is a workspace for interacting with the database, 
            # allowing you to query and manipulate database objects. By using `sessionmaker`, you can create a session factory that can be used to generate new session 
            # instances whenever needed.
            self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

            # Test the connection
            assert self.engine is not None
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")

            # Check which tables exist before creating
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()

            # Create tables if they don't exist (idempotent operation)
            Base.metadata.create_all(bind=self.engine)

            # Check if any new tables were created
            updated_tables = inspector.get_table_names()
            new_tables = set(updated_tables) - set(existing_tables)

            if new_tables:
                logger.info(f"Created new tables: {', '.join(new_tables)}")
            else:
                logger.info("All tables already exist - no new tables created")

            logger.info("PostgreSQL database initialized successfully")
            assert self.engine is not None
            logger.info(f"Database: {self.engine.url.database}")
            logger.info(f"Total tables: {', '.join(updated_tables) if updated_tables else 'None'}")
            logger.info("Database connection established")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise

    def teardown(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL database connections closed")

    @contextmanager
    ## what is @contextmanager?
    # The `@contextmanager` decorator is a convenient way to create context managers in Python. A context manager is a special type of object that defines the runtime context to 
    # be established when executing a `with` statement. It allows you to set up and tear down resources automatically, ensuring that resources are properly managed even if 
    # exceptions occur. By using `@contextmanager`, you can write a generator function that yields a resource (like a database session) and automatically handles the setup and 
    # cleanup logic around it.
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call startup() first.")

        session = self.session_factory()
        try:
            ## why yield session?
            # The `yield` statement in the `get_session` method allows it to be used as a context manager. When you call `get_session()`, it creates a new database session and 
            # yields it to the caller. The caller can then use this session to interact with the database. Once the caller is done with the session 
            # (i.e., after the block of code using the session is executed), control returns to the `get_session` method, which can then perform any necessary cleanup, such as 
            # rolling back transactions if an exception occurred or closing the session. This pattern ensures that database sessions are properly managed and resources are 
            # released even if errors occur during database operations.
            ## what if I used return instead of yield here?
            # If you used `return` instead of `yield` in the `get_session` method, it would not work as a context manager. The `return` statement would immediately exit the 
            # method and return the session object to the caller, but it would not allow for any cleanup logic to  be executed after the caller is done using the session. 
            # This means that if an exception occurs while using the session, there would be no opportunity to roll back any uncommitted transactions or close the session properly, 
            # which could lead to resource leaks and database integrity issues. Using `yield` allows the method to maintain control over the session lifecycle and ensure that 
            # resources are managed correctly.
            yield session
        except Exception:
            ## you mean if any exception occurs during the session usage, we want to roll back any uncommitted transactions to maintain database integrity? 
            session.rollback()
            raise
        finally:
            session.close()
