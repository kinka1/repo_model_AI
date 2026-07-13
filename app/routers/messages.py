from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models import InternalMessage, Specimen, User
from app.schemas import InternalMessageResponse, InternalMessageCreate

router = APIRouter(prefix="/api/messages", tags=["Pesan Internal"])


@router.get("/", response_model=List[InternalMessageResponse])
def get_messages(
    specimen_id: int = Query(..., description="ID spesimen untuk mengambil pesan"),
    db: Session = Depends(get_db),
):
    """
    Mengambil daftar pesan untuk suatu spesimen, diurutkan dari terlama ke terbaru.
    Semua role yang terautentikasi dapat membaca.
    """
    # Validasi spesimen exists
    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Spesimen tidak ditemukan")

    messages = (
        db.query(InternalMessage)
        .filter(InternalMessage.specimen_id == specimen_id)
        .order_by(InternalMessage.created_at.asc())
        .all()
    )

    result = []
    for msg in messages:
        sender = msg.sender
        result.append(InternalMessageResponse(
            id=msg.id,
            specimen_id=msg.specimen_id,
            sender_id=msg.sender_id,
            sender_name=sender.full_name if sender else "Unknown",
            sender_role=sender.role if sender else "Unknown",
            message_text=msg.message_text,
            created_at=msg.created_at,
        ))

    return result


@router.post("/", response_model=InternalMessageResponse, status_code=201)
def create_message(
    payload: InternalMessageCreate,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Mengirim pesan baru untuk suatu spesimen.
    sender_id diambil dari token autentikasi.
    """
    # Validasi spesimen exists
    specimen = db.query(Specimen).filter(Specimen.id == payload.specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Spesimen tidak ditemukan")

    sender_id = getattr(request.state, "user_id", None)
    if not sender_id:
        raise HTTPException(status_code=401, detail="User tidak terautentikasi")

    # Validasi user exists
    user = db.query(User).filter(User.id == sender_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    message = InternalMessage(
        specimen_id=payload.specimen_id,
        sender_id=sender_id,
        message_text=payload.message_text,
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    # Reload with relationship
    db.refresh(message)

    return InternalMessageResponse(
        id=message.id,
        specimen_id=message.specimen_id,
        sender_id=message.sender_id,
        sender_name=user.full_name,
        sender_role=user.role,
        message_text=message.message_text,
        created_at=message.created_at,
    )
