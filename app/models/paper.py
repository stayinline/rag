from sqlalchemy import Column, String, Text, Integer, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from app.models.base import Base, TimestampMixin, UUIDMixin


class Paper(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "papers"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    kb_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Basic metadata
    doi = Column(String(200), index=True)
    pmid = Column(String(50), index=True)
    title = Column(String(1000), nullable=False)
    authors = Column(Text)  # JSON list of {name, affiliation}
    journal = Column(String(300))
    publication_date = Column(DateTime(timezone=True))
    abstract = Column(Text)

    # Medical subject terms
    mesh_terms = Column(ARRAY(String(100)))  # MeSH terms
    diseases = Column(ARRAY(String(200)))
    drugs = Column(ARRAY(String(200)))
    targets = Column(ARRAY(String(200)))
    genes = Column(ARRAY(String(100)))

    # Study design
    study_type = Column(String(100))  # RCT, cohort, case-control, review, etc.
    sample_size = Column(Integer)
    has_randomization = Column(Integer)  # 0/1 boolean
    has_blinding = Column(Integer)  # 0/1 boolean

    # PICO
    pico_population = Column(Text)
    pico_intervention = Column(Text)
    pico_comparator = Column(Text)
    pico_outcome = Column(Text)

    # Evidence
    evidence_level = Column(String(50))  # A, B, C, etc.
    limitations = Column(Text)
    conclusion_strength = Column(String(50))  # strong, moderate, weak

    # References
    references = Column(JSONB, default=list)  # list of {doi, pmid, title, authors, year}
    cited_by = Column(JSONB, default=list)

    # Processing
    parser_version = Column(String(50))
    grobid_confidence = Column(Float)
    enhancement_source = Column(String(50))  # crossref, pubmed, mesh, none
    status = Column(String(50), default="draft")
