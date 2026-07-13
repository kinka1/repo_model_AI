import hashlib
import re
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
    RefreshRequest,
    RefreshResponse,
    ResetPasswordRequest,
)
from app.utils import get_local_now


router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# HL7 FHIR ISO 8601 duration format
# Examples: PT12H (12 hours), P7D (7 days), P30M (30 minutes)
FHIR_DURATION_RE = re.compile(
    r"^P"
    r"(?:(?P<years>\d+(?:\.\d+)?)Y)?"
    r"(?:(?P<months>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$"
)


def _parse_fhir_duration(fhir_str: str) -> timedelta:
    """Parse an HL7 FHIR ISO 8601 duration string into a timedelta.

    Examples:
        PT12H  → timedelta(hours=12)
        P7D    → timedelta(days=7)
        P30M   → timedelta(minutes=30)
        PT30M  → timedelta(minutes=30)
        P1DT6H → timedelta(days=1, hours=6)
    """
    m = FHIR_DURATION_RE.match(fhir_str)
    if not m:
        raise ValueError(f"Invalid FHIR duration format: {fhir_str!r}")

    parts = m.groupdict(default="0")
    return timedelta(
        days=float(parts["days"]),
        hours=float(parts["hours"]),
        minutes=float(parts["minutes"]),
        seconds=float(parts["seconds"]),
    )


SESSION_TTL = _parse_fhir_duration("PT12H")
REFRESH_TTL = _parse_fhir_duration("P7D")
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
    if session.expires_at < get_local_now():
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
    raw_refresh = secrets.token_urlsafe(48)
    now = get_local_now()

    expires_at = now + SESSION_TTL
    refresh_expires_at = now + REFRESH_TTL

    session = UserSession(
        user_id=user.id,
        token_hash=_hash_token(raw_token),
        expires_at=expires_at,
        is_revoked=False,
    )
    db.add(session)

    refresh_session = UserSession(
        user_id=user.id,
        token_hash=_hash_token(raw_refresh),
        expires_at=refresh_expires_at,
        is_revoked=False,
    )
    db.add(refresh_session)

    user.last_login = now
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
        refresh=raw_refresh,
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


@router.post("/refresh", response_model=RefreshResponse)
def refresh(
    payload: RefreshRequest,
    db: Session = Depends(get_db),
):
    raw_refresh = payload.refresh
    hashed = _hash_token(raw_refresh)
    now = get_local_now()

    refresh_session = (
        db.query(UserSession)
        .filter(
            UserSession.token_hash == hashed,
            UserSession.is_revoked.is_(False),
            UserSession.expires_at > now,
        )
        .first()
    )
    if not refresh_session:
        raise HTTPException(status_code=401, detail="Refresh token tidak valid atau sudah expired")

    # Revoke old access sessions for this user
    db.query(UserSession).filter(
        UserSession.user_id == refresh_session.user_id,
        UserSession.is_revoked.is_(False),
        UserSession.id != refresh_session.id,
    ).update({"is_revoked": True}, synchronize_session=False)

    # Issue new access token
    raw_token = secrets.token_urlsafe(48)
    expires_at = now + SESSION_TTL

    new_session = UserSession(
        user_id=refresh_session.user_id,
        token_hash=_hash_token(raw_token),
        expires_at=expires_at,
        is_revoked=False,
    )
    db.add(new_session)
    db.commit()

    return RefreshResponse(
        access_token=raw_token,
        expires_at=expires_at,
    )


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

    now = get_local_now()

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
    now = get_local_now()

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
