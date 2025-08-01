from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentInput(BaseModel):
    content: str  # Raw text extracted from the document
    filename: str
    doc_type: str = Field(..., description="Type of document: pdf, docx, email, etc.")
    metadata: Optional[dict] = None  # For extra info like sender, date, etc.

class HackRxInput(BaseModel):
    documents: List[DocumentInput]
    questions: List[str]

class ClauseMatch(BaseModel):
    clause_text: str
    clause_location: Optional[str] = None  # e.g., page/section
    score: Optional[float] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    rationale: str
    matched_clauses: List[ClauseMatch]