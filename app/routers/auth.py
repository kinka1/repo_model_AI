import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import PasswordResetToken
from app.models import Session as UserSession
from app.models import User
from app.schemas import (
    AuthMessageResponse,
    AuthUserResponse,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LoginRequest,
    LoginResponse,
    ResetPasswordRequest,
)


router = APIRouter(prefix="/api/auth", tags=["Authentication"])

SESSION_TTL_HOURS = 12
RESET_TOKEN_TTL_MINUTES = 30


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header dibutuhkan")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Format Authorization harus Bearer <token>")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token tidak boleh kosong")
    return token


def _extract_bearer_token_from_request(request: Request) -> str:
    authorization = request.headers.get("Authorization")
    return _extract_bearer_token(authorization)


def _find_valid_session(db: Session, raw_token: str) -> UserSession:
    token_hash = _hash_token(raw_token)
    session = (
        db.query(UserSession)
        .filter(UserSession.token_hash == token_hash, UserSession.is_revoked.is_(False))
        .first()
    )
    if not session:
        raise HTTPException(status_code=401, detail="Token tidak valid")
    if session.expires_at < datetime.utcnow():
        session.is_revoked = True
        db.add(session)
        db.commit()
        raise HTTPException(status_code=401, detail="Token sudah expired")
    return session


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Username atau password salah")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User nonaktif")

    incoming = _hash_password(payload.password)
    if incoming != user.hashed_password:
        raise HTTPException(status_code=401, detail="Username atau password salah")

    raw_token = secrets.token_urlsafe(48)
    expires_at = datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)

    session = UserSession(
        user_id=user.id,
        token_hash=_hash_token(raw_token),
        expires_at=expires_at,
        is_revoked=False,
    )
    db.add(session)

    user.last_login = datetime.utcnow()
    db.add(user)
    db.commit()

    return LoginResponse(
        access_token=raw_token,
        expires_at=expires_at,
        user=AuthUserResponse(
            id=user.id,
            full_name=user.full_name,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
        ),
    )


@router.get("/me", response_model=AuthUserResponse)
def me(
    request: Request,
    db: Session = Depends(get_db),
):
    token = _extract_bearer_token_from_request(request)
    session = _find_valid_session(db, token)
    user = db.query(User).filter(User.id == session.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User tidak ditemukan")

    return AuthUserResponse(
        id=user.id,
        full_name=user.full_name,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
    )


@router.post("/logout", response_model=AuthMessageResponse)
def logout(
    request: Request,
    db: Session = Depends(get_db),
):
    token = _extract_bearer_token_from_request(request)
    session = _find_valid_session(db, token)
    session.is_revoked = True
    db.add(session)
    db.commit()
    return AuthMessageResponse(message="Logout berhasil")


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = (
        db.query(User)
        .filter(
            or_(
                User.username == payload.username_or_email,
                User.email == payload.username_or_email,
            )
        )
        .first()
    )

    generic_message = "Jika akun ditemukan, token reset password sudah dibuat"
    if not user:
        return ForgotPasswordResponse(message=generic_message)

    now = datetime.utcnow()

    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.is_used.is_(False),
        PasswordResetToken.expires_at > now,
    ).update({"is_used": True}, synchronize_session=False)

    raw_token = secrets.token_urlsafe(40)
    expires_at = now + timedelta(minutes=RESET_TOKEN_TTL_MINUTES)

    reset = PasswordResetToken(
        user_id=user.id,
        token_hash=_hash_token(raw_token),
        expires_at=expires_at,
        is_used=False,
    )
    db.add(reset)
    db.commit()

    return ForgotPasswordResponse(
        message=generic_message,
        reset_token=raw_token,
        expires_at=expires_at,
    )


@router.post("/reset-password", response_model=AuthMessageResponse)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_hash = _hash_token(payload.token)
    now = datetime.utcnow()

    reset = (
        db.query(PasswordResetToken)
        .filter(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.is_used.is_(False),
        )
        .first()
    )

    if not reset:
        raise HTTPException(status_code=400, detail="Token reset tidak valid")
    if reset.expires_at < now:
        reset.is_used = True
        db.add(reset)
        db.commit()
        raise HTTPException(status_code=400, detail="Token reset sudah expired")

    user = db.query(User).filter(User.id == reset.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    user.hashed_password = _hash_password(payload.new_password)
    reset.is_used = True

    db.query(UserSession).filter(
        UserSession.user_id == user.id,
        UserSession.is_revoked.is_(False),
    ).update({"is_revoked": True}, synchronize_session=False)

    db.add(user)
    db.add(reset)
    db.commit()

    return AuthMessageResponse(message="Password berhasil direset")


@router.put("/change-password", response_model=AuthMessageResponse)
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    token = _extract_bearer_token_from_request(request)
    session = _find_valid_session(db, token)

    user = db.query(User).filter(User.id == session.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    user.hashed_password = _hash_password(payload.new_password)

    # Revoke semua sesi lama (termasuk sesi saat ini) agar user login ulang.
    db.query(UserSession).filter(
        UserSession.user_id == user.id,
        UserSession.is_revoked.is_(False),
    ).update({"is_revoked": True}, synchronize_session=False)

    db.add(user)
    db.commit()

    return AuthMessageResponse(message="Password berhasil diganti, silakan login ulang")
