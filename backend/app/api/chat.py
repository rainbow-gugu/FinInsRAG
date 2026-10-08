"""对话与会话接口：SSE 流式对话 / 会话管理 / 消息记录"""
import json
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.core.security import get_current_user
from app.models.schemas import (
    ChatRequest,
    CreateSessionResponse,
    MessageRecord,
    SessionInfo,
)
from app.rag.pipeline import get_pipeline

router = APIRouter(tags=["chat"])

# 内存会话存储
_sessions: Dict[str, Dict] = {}
_messages: Dict[str, List[Dict]] = {}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@router.get("/get_sessions", response_model=List[SessionInfo])
def list_sessions(user=Depends(get_current_user)):
    """获取当前用户的会话列表"""
    sessions = [
        SessionInfo(
            session_id=s["session_id"],
            title=s.get("title", "新对话"),
            created_at=s["created_at"],
            message_count=len(_messages.get(s["session_id"], [])),
        )
        for s in _sessions.values()
        if s.get("user_id") == user.user_id
    ]
    return sorted(sessions, key=lambda x: x.created_at, reverse=True)


@router.post("/create_session", response_model=CreateSessionResponse)
def create_session(user=Depends(get_current_user)):
    """创建新会话"""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id": session_id,
        "user_id": user.user_id,
        "title": "新对话",
        "created_at": _now(),
    }
    _messages[session_id] = []
    return CreateSessionResponse(session_id=session_id)


@router.get("/get_messages", response_model=List[MessageRecord])
def get_messages(
    session_id: str = Query(...),
    user=Depends(get_current_user),
):
    """获取指定会话的消息记录"""
    msgs = _messages.get(session_id, [])
    return [
        MessageRecord(
            message_id=m["message_id"],
            session_id=m["session_id"],
            user_question=m["user_question"],
            model_answer=m["model_answer"],
            think=m.get("think"),
            documents=m.get("documents"),
            recommended_questions=m.get("recommended_questions"),
            created_at=m["created_at"],
        )
        for m in msgs
    ]


@router.get("/sessions/{session_id}/documents")
def get_session_documents(session_id: str, user=Depends(get_current_user)):
    """获取会话关联的文档（简化版：返回全局知识库文档）"""
    pipeline = get_pipeline()
    docs = pipeline.list_documents()
    return {
        "documents": [
            {
                "id": i,
                "session_id": session_id,
                "document_name": d["doc_name"],
                "document_type": d.get("doc_type", ""),
                "file_size": 0,
                "created_at": _now(),
                "updated_at": _now(),
                "upload_time": _now(),
            }
            for i, d in enumerate(docs)
        ],
        "has_documents": len(docs) > 0,
    }


@router.post("/chat_on_docs")
async def chat_on_docs(
    req: ChatRequest,
    session_id: str = Query(...),
    user=Depends(get_current_user),
):
    """
    基于知识库的流式对话（SSE）。
    事件类型: references / thinking / content / done / error
    """
    pipeline = get_pipeline()

    async def event_stream() -> AsyncGenerator[str, None]:
        full_answer = ""
        full_thinking = ""
        references_data = []

        def _fmt_docs(refs):
            return [
                {
                    "document_id": r.get("chunk_id", ""),
                    "document_name": r.get("doc_name", ""),
                    "content_with_weight": r.get("content", ""),
                    "score": r.get("score", r.get("final_score", 0)),
                }
                for r in refs
            ]

        for event in pipeline.chat_stream(
            req.message,
            session_id=session_id,
            filter=req.filter.model_dump() if req.filter else None,
        ):
            etype = event.get("type")

            if etype == "references":
                references_data = event["data"].get("references", [])
                frame = {"documents": _fmt_docs(references_data)}
                yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"

            elif etype == "thinking":
                full_thinking += event.get("text", "")
                frame = {"content": event.get("text", ""), "thinking": True}
                yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"

            elif etype == "content":
                full_answer += event.get("text", "")
                frame = {"content": event.get("text", ""), "thinking": False}
                yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"

            elif etype == "done":
                data = event.get("data", {})
                full_answer = data.get("answer", full_answer)
                full_thinking = data.get("thinking", full_thinking)
                recommended = data.get("recommended_questions", [])
                end_frame = {
                    "content": full_answer,
                    "thinking": False,
                    "think": full_thinking,
                    "documents": _fmt_docs(references_data),
                    "recommended_questions": recommended,
                }
                yield f"data: {json.dumps(end_frame, ensure_ascii=False)}\n\n"

            elif etype == "error":
                yield f"data: {json.dumps({'error': event.get('text', '')}, ensure_ascii=False)}\n\n"

        # 保存消息记录
        msg_id = str(uuid.uuid4())
        if session_id not in _messages:
            _messages[session_id] = []
        _messages[session_id].append(
            {
                "message_id": msg_id,
                "session_id": session_id,
                "user_question": req.message,
                "model_answer": full_answer,
                "think": full_thinking,
                "documents": json.dumps(_fmt_docs(references_data), ensure_ascii=False),
                "recommended_questions": json.dumps(
                    [], ensure_ascii=False
                ),
                "created_at": _now(),
            }
        )

        # 更新会话标题（用第一条用户消息）
        if session_id in _sessions and _sessions[session_id].get("title") == "新对话":
            _sessions[session_id]["title"] = req.message[:20]

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/quick_parse")
async def quick_parse(
    session_id: str = Query(...),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """快速解析文件并摄入到当前会话知识库"""
    import os
    from app.core.config import settings
    from app.rag.pipeline import get_pipeline

    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"quick_{session_id}_{file.filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        pipeline = get_pipeline()
        result = pipeline.ingest_file(file_path, user_id=user.user_id)
        return {"status": "success", "session_id": session_id, **result}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
