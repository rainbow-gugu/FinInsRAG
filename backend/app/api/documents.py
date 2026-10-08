"""知识库文档接口：上传 / 列表 / 删除"""
import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.security import get_current_user
from app.models.schemas import DeleteResponse, DocumentInfo, UploadResponse
from app.rag.pipeline import get_pipeline

router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}


@router.get("/get_files", response_model=List[DocumentInfo])
def list_documents(user=Depends(get_current_user)):
    """获取知识库中的文档列表"""
    import time

    pipeline = get_pipeline()
    docs = pipeline.list_documents()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return [
        DocumentInfo(
            file_name=d["doc_name"],
            doc_type=d.get("doc_type", ""),
            chunk_count=d["chunk_count"],
            created_at=now,
            updated_at=now,
            user_id=user.user_id,
        )
        for d in docs
    ]


@router.post("/upload_files", response_model=UploadResponse)
async def upload_document(
    files: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """上传文档并自动摄入到知识库"""
    ext = os.path.splitext(files.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # 保存上传文件
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{files.filename}"
    file_path = os.path.join(upload_dir, safe_name)

    content = await files.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过限制 (50MB)")

    with open(file_path, "wb") as f:
        f.write(content)

    # 摄入到 RAG 流水线
    try:
        pipeline = get_pipeline()
        result = pipeline.ingest_file(file_path, user_id=user.user_id)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")

    return UploadResponse(
        message="文档上传并解析成功",
        file_id=file_id,
        doc_name=result["doc_name"],
        doc_type=result["doc_type"],
        total_chunks=result["total_chunks"],
        char_count=result["char_count"],
    )


@router.delete("/delete_file/{file_name}", response_model=DeleteResponse)
def delete_document(file_name: str, user=Depends(get_current_user)):
    """按文档名删除知识库中的文档"""
    pipeline = get_pipeline()
    deleted = pipeline.delete_document(file_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DeleteResponse(deleted=deleted)
