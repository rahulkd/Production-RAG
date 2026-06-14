from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from src.models.paper import Paper
from src.schemas.arxiv.paper import PaperCreate

## explain the purpose of this file?
# The `paper.py` file in the `repositories` directory serves as the data access layer for the `Paper` model. It defines the `PaperRepository` class, which provides methods for
# performing CRUD (Create, Read, Update, Delete) operations on `Paper` objects in the database. 
#  This repository abstracts away the details of database interactions, allowing other parts of the application to  interact with `Paper` data without needing to know about the 
#  underlying database implementation.
class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, paper: PaperCreate) -> Paper:
        ## what is this method doing?
        # The `create` method takes a `PaperCreate` schema object as input, which contains the data needed to create a new `Paper` record in the database. 
        # It converts the `PaperCreate` object into a `Paper` SQLAlchemy model instance, adds it to the database session and commits the transaction to save the new record in the database. 
        # After committing, it refreshes the instance to get any database-generated fields (like `id` or timestamps) and returns the newly created `Paper` object.
        db_paper = Paper(**paper.model_dump())
        self.session.add(db_paper)
        self.session.commit()  ## does this mean the paper is stored in the database?
        ## yes, calling `commit()` on the session will persist the changes made to the database, including adding the new `Paper` record. Once `commit()` is called, 
        # the new record is stored in the database and can be retrieved in subsequent queries.
        self.session.refresh(db_paper)
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        return self.session.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()

    def get_by_id(self, paper_id: UUID) -> Optional[Paper]:
        return self.session.query(Paper).filter(Paper.id == paper_id).first()

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        return self.session.query(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset).all()

    def update(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def upsert(self, paper_create: PaperCreate) -> Paper:
        ## what is this method doing?
        # The `upsert` method is a combination of "update" and "insert". It first checks if a `Paper` record with the given `arxiv_id` already exists in the database. 
        # If it does exist, it updates the existing record with the new data from the `PaperCreate` object. Otherwise, it creates a new record.
        # Check if paper already exists
        existing_paper = self.get_by_arxiv_id(paper_create.arxiv_id)
        if existing_paper:
            # Update existing paper
            for key, value in paper_create.model_dump(exclude_unset=True).items():
                setattr(existing_paper, key, value)
            return self.update(existing_paper)
        else:
            # Create new paper
            return self.create(paper_create)
