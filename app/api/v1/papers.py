"""Paper Intelligence API endpoints."""
import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models.document import Document, DocumentVersion
from app.models.paper import Paper
from app.models.task import IngestionJob
from app.schemas.paper import (
    DoiImportRequest,
    PmidImportRequest,
    PaperEvidenceResponse,
    PaperReferencesResponse,
    PaperResponse,
    PaperUploadResponse,
    SimilarPapersResponse,
    SimilarPaperItem,
)

logger = logging.getLogger(__name__)
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
    logger.info(
        "Upload paper request org_id=%s user_id=%s kb_id=%s filename=%s doi=%s pmid=%s content_type=%s",
        org_id,
        user["user_id"],
        kb_id,
        file.filename,
        doi,
        pmid,
        file.content_type,
    )

    # Save uploaded file
    os.makedirs(os.path.join(settings.storage_path, org_id, kb_id), exist_ok=True)
    filename = f"{uuid.uuid4()}_{file.filename}"
    storage_path = os.path.join(settings.storage_path, org_id, kb_id, filename)

    with open(storage_path, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info(
        "Upload paper file saved org_id=%s user_id=%s kb_id=%s filename=%s size_bytes=%s storage_path=%s",
        org_id,
        user["user_id"],
        kb_id,
        file.filename,
        len(content),
        storage_path,
    )

    document_id = uuid.uuid4()
    version_id = uuid.uuid4()

    # Create Document record
    session = db
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
        embedding_model=settings.embedding_model,
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
    logger.info(
        "Upload paper records created org_id=%s user_id=%s kb_id=%s paper_id=%s document_id=%s version_id=%s job_id=%s",
        org_id,
        user["user_id"],
        kb_id,
        paper_id,
        document_id,
        version_id,
        job_id,
    )

    # Kick off Celery task
    from app.workers.celery_app import celery_app
    celery_app.send_task(
        "parse_paper",
        args=[org_id, str(document_id), str(version_id), str(paper_id), storage_path, doi, pmid, kb_id],
    )
    logger.info(
        "Upload paper parse task queued org_id=%s user_id=%s paper_id=%s document_id=%s version_id=%s job_id=%s",
        org_id,
        user["user_id"],
        paper_id,
        document_id,
        version_id,
        job_id,
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

    logger.info("Import DOI request org_id=%s user_id=%s doi=%s kb_id=%s", user["org_id"], user["user_id"], data.doi, data.kb_id)
    metadata = enhance_via_crossref(data.doi)
    if not metadata:
        logger.warning("Import DOI failed org_id=%s user_id=%s doi=%s reason=not_found", user["org_id"], user["user_id"], data.doi)
        raise HTTPException(status_code=404, detail=f"Paper with DOI {data.doi} not found via CrossRef")

    logger.info(
        "Import DOI succeeded org_id=%s user_id=%s doi=%s has_title=%s author_count=%s",
        user["org_id"],
        user["user_id"],
        data.doi,
        bool(metadata.title),
        len(metadata.authors),
    )
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

    logger.info("Import PMID request org_id=%s user_id=%s pmid=%s kb_id=%s", user["org_id"], user["user_id"], data.pmid, data.kb_id)
    metadata = enhance_via_pubmed(data.pmid)
    if not metadata:
        logger.warning("Import PMID failed org_id=%s user_id=%s pmid=%s reason=not_found", user["org_id"], user["user_id"], data.pmid)
        raise HTTPException(status_code=404, detail=f"Paper with PMID {data.pmid} not found via PubMed")

    logger.info(
        "Import PMID succeeded org_id=%s user_id=%s pmid=%s has_title=%s author_count=%s mesh_count=%s",
        user["org_id"],
        user["user_id"],
        data.pmid,
        bool(metadata.title),
        len(metadata.authors),
        len(metadata.mesh_terms),
    )
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
    logger.info("Get paper request org_id=%s user_id=%s paper_id=%s", user["org_id"], user["user_id"], paper_id)
    session = db
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        logger.warning("Get paper failed org_id=%s user_id=%s paper_id=%s reason=not_found", user["org_id"], user["user_id"], paper_id)
        raise HTTPException(status_code=404, detail="Paper not found")
    logger.info("Get paper succeeded org_id=%s user_id=%s paper_id=%s status=%s", user["org_id"], user["user_id"], paper_id, paper.status)
    return paper


@router.get("/{paper_id}/evidence", response_model=PaperEvidenceResponse)
async def get_paper_evidence(
    paper_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get PICO / evidence summary for a paper."""
    logger.info("Get paper evidence request org_id=%s user_id=%s paper_id=%s", user["org_id"], user["user_id"], paper_id)
    session = db
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        logger.warning("Get paper evidence failed org_id=%s user_id=%s paper_id=%s reason=not_found", user["org_id"], user["user_id"], paper_id)
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
    logger.info("Get paper references request org_id=%s user_id=%s paper_id=%s", user["org_id"], user["user_id"], paper_id)
    session = db
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        logger.warning("Get paper references failed org_id=%s user_id=%s paper_id=%s reason=not_found", user["org_id"], user["user_id"], paper_id)
        raise HTTPException(status_code=404, detail="Paper not found")

    from app.schemas.paper import PaperReference
    refs = paper.references or []
    logger.info("Get paper references succeeded org_id=%s user_id=%s paper_id=%s total=%s", user["org_id"], user["user_id"], paper_id, len(refs))
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
    logger.info("Get similar papers request org_id=%s user_id=%s paper_id=%s", user["org_id"], user["user_id"], paper_id)
    session = db
    paper = await session.get(Paper, paper_id)
    if not paper or str(paper.org_id) != str(user["org_id"]):
        logger.warning("Get similar papers failed org_id=%s user_id=%s paper_id=%s reason=not_found", user["org_id"], user["user_id"], paper_id)
        raise HTTPException(status_code=404, detail="Paper not found")

    # Find papers with overlapping MeSH terms in same org
    mesh_terms = paper.mesh_terms or []
    if not mesh_terms:
        logger.info("Get similar papers complete org_id=%s user_id=%s paper_id=%s reason=no_mesh_terms", user["org_id"], user["user_id"], paper_id)
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
    logger.info(
        "Get similar papers complete org_id=%s user_id=%s paper_id=%s candidates=%s similar=%s",
        user["org_id"],
        user["user_id"],
        paper_id,
        len(candidates),
        len(similar[:5]),
    )
    return SimilarPapersResponse(paper_id=paper_id, similar_papers=similar[:5])
