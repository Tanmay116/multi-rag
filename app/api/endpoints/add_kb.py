from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.services.ingestion.pdf import ingest_document_flow
# from app.services.ingestion.web import ingest_webpage_flow

ingest_router = APIRouter()


@ingest_router.post("/pdf")
async def upload_file(
    file: UploadFile = File(...), description: str = Header(..., alias="X-Description")
):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename detected")

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400, detail="Only PDF and TXT files are supported"
        )

    result = await ingest_document_flow(description, file)
    return {"status": result}

# @ingest_router.post("/pdf")
# async def upload_webpage(
#     webpage: str,  description: str = Header(..., alias="X-Description")
# ):
#     if not webpage:
#         raise HTTPException(status_code=400, detail="No weblink given")

#     result = await ingest_document_flow(description, file)
#     return {"status": result}
