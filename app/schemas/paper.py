from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PaperAuthor(BaseModel):
    name: str
    affiliation: str | None = None


class PaperReference(BaseModel):
    doi: str | None = None
    pmid: str | None = None
    title: str | None = None
    authors: str | None = None
    year: int | None = None


class PaperCreate(BaseModel):
    kb_id: UUID
    title: str = Field(..., min_length=1, max_length=1000)
    doi: str | None = None
    pmid: str | None = None


class PaperUploadResponse(BaseModel):
    paper_id: UUID
    document_id: UUID
    title: str
    status: str
    ingestion_job_id: UUID


class PaperMetadata(BaseModel):
    doi: str | None = None
    pmid: str | None = None
    title: str | None = None
    authors: list[PaperAuthor] = Field(default_factory=list)
    journal: str | None = None
    publication_date: datetime | None = None
    abstract: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    diseases: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list)
    study_type: str | None = None
    sample_size: int | None = None


class PaperResponse(BaseModel):
    id: UUID
    org_id: UUID
    kb_id: UUID
    document_id: UUID
    doi: str | None = None
    pmid: str | None = None
    title: str
    authors: str | None = None
    journal: str | None = None
    publication_date: datetime | None = None
    abstract: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    diseases: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    genes: list[str] = Field(default_factory=list)
    study_type: str | None = None
    sample_size: int | None = None
    has_randomization: bool = False
    has_blinding: bool = False
    pico_population: str | None = None
    pico_intervention: str | None = None
    pico_comparator: str | None = None
    pico_outcome: str | None = None
    evidence_level: str | None = None
    limitations: str | None = None
    conclusion_strength: str | None = None
    references: list[dict] = Field(default_factory=list)
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PaperEvidenceResponse(BaseModel):
    paper_id: UUID
    title: str
    study_type: str | None = None
    sample_size: int | None = None
    has_randomization: bool = False
    has_blinding: bool = False
    pico: dict | None = None
    evidence_level: str | None = None
    conclusion_strength: str | None = None
    limitations: str | None = None


class PaperReferencesResponse(BaseModel):
    paper_id: UUID
    title: str
    references: list[PaperReference]
    total: int


class SimilarPaperItem(BaseModel):
    paper_id: UUID
    title: str
    doi: str | None = None
    similarity_score: float
    shared_mesh: list[str] = Field(default_factory=list)


class SimilarPapersResponse(BaseModel):
    paper_id: UUID
    similar_papers: list[SimilarPaperItem]


class DoiImportRequest(BaseModel):
    doi: str = Field(..., min_length=1)
    kb_id: UUID


class PmidImportRequest(BaseModel):
    pmid: str = Field(..., min_length=1)
    kb_id: UUID
