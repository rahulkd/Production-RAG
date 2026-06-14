from abc import ABC, abstractmethod
from typing import Any, ContextManager, Dict, List, Optional

from sqlalchemy.orm import Session

## what is this file for?
# The `base.py` file in the `db/interfaces` directory defines abstract base classes for database operations and repositories. 
# The `BaseDatabase` class provides an interface for initializing, tearing down, and managing database sessions, while the `BaseRepository` class defines a standard interface 
# for CRUD operations on database records. 
# These base classes serve as templates for concrete implementations, ensuring that all database interactions follow a consistent pattern across the application. 
# By using abstract base classes, we can enforce a contract for how databases and repositories should behave, making it easier to maintain and extend the application in the future.
class BaseDatabase(ABC):
    """Base class for database operations."""

    @abstractmethod
    def startup(self) -> None:
        """Initialize the database connection."""

    @abstractmethod
    def teardown(self) -> None:
        """Close the database connection."""

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        """Get a database session."""


class BaseRepository(ABC):
    """Base repository pattern for data access."""

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record."""

    @abstractmethod
    def get_by_id(self, record_id: Any) -> Optional[Any]:
        """Get a record by ID."""

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        """Update a record by ID."""

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record by ID."""

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List records with pagination."""
