"""Pydantic 数据模型"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------- 认证 ----------

class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserInfo(BaseModel):
    user_id: str


# ---------- 文档 ----------

class DocumentInfo(BaseModel):
    file_name: str
    doc_type: str = ""
    chunk_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    user_id: str = ""
    status: str = "ready"


class UploadResponse(BaseModel):
    status: str = "success"
    message: str
    file_id: str
    doc_name: str
    doc_type: str
    total_chunks: int
    char_count: int


class DeleteResponse(BaseModel):
    status: str = "success"
    deleted: int


# ---------- 会话 ----------

class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    message_count: int = 0


class CreateSessionResponse(BaseModel):
    status: str = "success"
    session_id: str


class MessageRecord(BaseModel):
    message_id: str
    session_id: str
    user_question: str
    model_answer: str
    think: Optional[str] = None
    documents: Optional[str] = None
    recommended_questions: Optional[str] = None
    created_at: str


# ---------- 对话 ----------

class MetadataFilter(BaseModel):
    """元数据前置过滤条件（全部可选，AND 关系）"""
    doc_name: Optional[str] = None   # 限定文档名
    doc_type: Optional[str] = None   # 限定文档类型：pdf/docx/xlsx
    kb_id: Optional[str] = None     # 限定知识库


class ChatRequest(BaseModel):
    message: str
    filter: Optional[MetadataFilter] = None


# ---------- 反馈 ----------

class FeedbackCreate(BaseModel):
    message_id: str
    rating: int          # 1=有帮助, -1=没帮助
    reason: str = ""     # 可选：错误原因或改进建议
