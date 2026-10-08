"""用户反馈接口：对回答点赞/点踩，收集改进数据（人类在环闭环）"""
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, Depends

from app.core.security import get_current_user
from app.models.schemas import FeedbackCreate

router = APIRouter(tags=["feedback"])

# 内存反馈存储（生产环境换数据库；与项目现有内存存储风格一致）
_feedback: List[Dict] = []


@router.post("/feedback")
def submit_feedback(req: FeedbackCreate, user=Depends(get_current_user)):
    """提交对某条回答的反馈：rating=1 有帮助, -1 没帮助"""
    record = {
        "user_id": user.user_id,
        "message_id": req.message_id,
        "rating": req.rating,
        "reason": req.reason,
        "created_at": datetime.now().isoformat(),
    }
    _feedback.append(record)
    return {"status": "success", "message": "反馈已提交，感谢你的帮助"}


@router.get("/feedback/stats")
def feedback_stats(user=Depends(get_current_user)):
    """查看反馈统计（有帮助率）"""
    total = len(_feedback)
    helpful = sum(1 for f in _feedback if f["rating"] == 1)
    not_helpful = sum(1 for f in _feedback if f["rating"] == -1)
    return {
        "total": total,
        "helpful": helpful,
        "not_helpful": not_helpful,
        "helpful_rate": helpful / total if total else 0,
    }
