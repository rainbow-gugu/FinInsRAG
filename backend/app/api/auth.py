"""认证接口：注册 / 登录 / 当前用户"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.models.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserInfo,
)

router = APIRouter(tags=["auth"])

# 内存用户存储（生产环境换数据库）
_users: dict = {}


@router.post("/register", response_model=dict)
def register(req: RegisterRequest):
    if req.username in _users:
        raise HTTPException(status_code=400, detail="用户已存在")
    user_id = f"u_{req.username}"
    _users[req.username] = {
        "user_id": user_id,
        "password_hash": hash_password(req.password),
    }
    return {"status": "success", "user_id": user_id}


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = _users.get(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    token = create_access_token(user["user_id"])
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserInfo)
def read_me(user=Depends(get_current_user)):
    return UserInfo(user_id=user.user_id)
