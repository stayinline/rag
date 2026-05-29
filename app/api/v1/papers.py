"""Paper Intelligence API endpoints."""
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user
from app.database import get_db
from app.models.document import Document, DocumentVersion
from app.models.paper import Paper
from app.models.task import IngestionJob
from app.schemas.paper import (
    DoiImportRequest,
    PmidImportRequest,
    PaperCreate,
    PaperEvidenceResponse,
    PaperReferencesResponse,
    PaperResponse,
    PaperUploadResponse,
    SimilarPapersResponse,
    SimilarPaperItem,
)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("/upload", response_model=PaperUploadResponse)
async def upload_paper(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    doi: str | None = Form(None),
    pmid: str | None = Form(None),
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Upload a SCI PDF paper for parsing and indexing."""
    org_id = str(user["org_id"])

    # Save uploaded file
    os.makedirs(os.path.join("./data/files", org_id, kb_id), exist_ok=True)
    filename = f"{uuid.uuid4()}_{file.filename}"
    storage_path = os.path.join("./data/files", org_id, kb_id, filename)

    with open(storage_path, "wb") as f:
        content = await file.read()
        f.write(content)

    document_id = uuid.uuid4()
    version_id = uuid.uuid4()

    # Create Document record
    session = await db.__anext__()
    doc = Document(
        id=document_id,
        org_id=org_id,
        kb_id=kb_id,
        title=doi or file.filename or "Untitled Paper",
        file_name=file.filename,
        file_type="pdf",
        source_type="doi" if doi else ("pmid" if pmid else "upload"),
        source_uri=f"https://doi.org/{doi}" if doi else None,
        current_version=1,
        status="draft",
        security_level="internal",
        document_type="paper",
        created_by=user["user_id"],
    )
    session.add(doc)

    # Create DocumentVersion
    version = DocumentVersion(
        id=version_id,
        org_id=org_id,
        document_id=document_id,
        version=1,
        storage_path=storage_path,
        parser_version="paper_parser_v1",
        embedding_model="text-embedding-v3",
        index_status="pending",
    )
    session.add(version)

    # Create Paper record
    paper_id = uuid.uuid4()
    paper = Paper(
        id=paper_id,
        org_id=org_id,
        kb_id=kb_id,
        document_id=document_id,
        doi=doi,
        pmid=pmid,
        title=doi or file.filename or "Untitled Paper",
        status="draft",
    )
    session.add(paper)

    # Create ingestion job
    job_id = uuid.uuid4()
    job = IngestionJob(
        id=job_id,
        org_id=org_id,
        document_id=document_id,
        version_id=version_id,
        job_type="parse",
        status="pending",
        idempotency_key=f"{org_id}:{document_id}:{version_id}:parse_paper",
    )
    session.add(job)
    await session.commit()

    # Kick off Celery task
    from app.workers.celery_app import celery_app
    celery_app.send_task(
        "parse_paper_task",
        args=[org_id, str(document_id), str(version_id), str(paper_id), storage_path, doi, pmid],
    )

    return PaperUploadResponse(
        paper_id=paper_id,
        document_id=document_id,
        title=doc.title,
        status=doc.status,
        ingestion_job_id=job_id,
    )


@router.post("/import-doi")
async def import_by_doi(
    data: DoiImportRequest,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Import paper metadata via DOI from CrossRef."""
    from app.services.metadata_enhancer import enhance_via_crossref

    metadata = enhance_via_crossref(data.doi)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Paper with DOI {data.doi} not found via CrossRef")

    return {
        "doi": data.doi,
        "title": metadata.title,
        "authors": metadata.authors,
        "journal": metadata.journal,
        "publication_date": metadata.publication_date,
        "abstract": metadata.abstract,
    }


@router.post("/import-pmid")
async def import_by_pmid(
    data: PmidImportRequest,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Import paper metadata via PMID from PubMed."""
    from app.services.metadata_enhancer import enhance_via_pubmed

    metadata = enhance_via_pubmed(data.pmid)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Paper with PMID {data.pmid} not found via PubMed")

    return {
        "pmid": data.pmid,
        "title": metadata.title,
        "authors": metadata.authors,
        "journal": metadata.journal,
        "publication_date": metadata.publication_date,
        "abstract": metadata.abstract,
        "mesh_terms": metadata.mesh_terms,
    }


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(
    paper_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get structured paper details."""
    session = await db.__anext__()
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.get("/{paper_id}/evidence", response_model=PaperEvidenceResponse)
async def get_paper_evidence(
    paper_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get PICO / evidence summary for a paper."""
    session = await db.__anext__()
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        raise HTTPException(status_code=404, detail="Paper not found")

    return PaperEvidenceResponse(
        paper_id=paper.id,
        title=paper.title,
        study_type=paper.study_type,
        sample_size=paper.sample_size,
        has_randomization=bool(paper.has_randomization),
        has_blinding=bool(paper.has_blinding),
        pico={
            "population": paper.pico_population,
            "intervention": paper.pico_intervention,
            "comparator": paper.pico_comparator,
            "outcome": paper.pico_outcome,
        } if paper.pico_population or paper.pico_intervention else None,
        evidence_level=paper.evidence_level,
        conclusion_strength=paper.conclusion_strength,
        limitations=paper.limitations,
    )


@router.get("/{paper_id}/references", response_model=PaperReferencesResponse)
async def get_paper_references(
    paper_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get references cited by a paper."""
    session = await db.__anext__()
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        raise HTTPException(status_code=404, detail="Paper not found")

    from app.schemas.paper import PaperReference
    refs = paper.references or []
    return PaperReferencesResponse(
        paper_id=paper.id,
        title=paper.title,
        references=[PaperReference(**r) for r in refs if isinstance(r, dict)],
        total=len(refs),
    )


@router.get("/{paper_id}/similar", response_model=SimilarPapersResponse)
async def get_similar_papers(
    paper_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Find similar papers based on MeSH terms and domain tags."""
    session = await db.__anext__()
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        raise HTTPException(status_code=404, detail="Paper not found")

    # Find papers with overlapping MeSH terms in same org
    mesh_terms = paper.mesh_terms or []
    if not mesh_terms:
        return SimilarPapersResponse(paper_id=paper_id, similar_papers=[])

    stmt = (
        select(Paper)
        .where(
            Paper.org_id == paper.org_id,
            Paper.id != paper_id,
            Paper.status == "ready",
        )
        .limit(5)
    )
    result = await session.execute(stmt)
    candidates = result.scalars().all()

    similar = []
    for candidate in candidates:
        candidate_mesh = set(candidate.mesh_terms or [])
        shared = list(set(mesh_terms) & candidate_mesh)
        if shared:
            score = len(shared) / max(len(set(mesh_terms) | candidate_mesh), 1)
            similar.append(SimilarPaperItem(
                paper_id=candidate.id,
                title=candidate.title,
                doi=candidate.doi,
                similarity_score=round(score, 3),
                shared_mesh=shared[:5],
            ))

    similar.sort(key=lambda x: x.similarity_score, reverse=True)
    return SimilarPapersResponse(paper_id=paper_id, similar_papers=similar[:5])
