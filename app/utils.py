import math
from typing import Any, Tuple
from sqlalchemy.orm import Query
from .schemas import PaginationMeta

def paginate_query(query: Query, page: int, per_page: int) -> Tuple[Any, PaginationMeta]:
    """
    Helper untuk melakukan pagination pada query SQLAlchemy.
    Mengembalikan tuple (data_list, pagination_meta).
    """
    total = query.count()
    
    # Pastikan per_page minimal 1
    if per_page < 1: per_page = 1
    
    last_page = math.ceil(total / per_page) if total > 0 else 1
    
    # Normalisasi halaman agar tidak out of bounds
    if page < 1: 
        page = 1
    elif page > last_page and last_page > 0:
        page = last_page
        
    offset = (page - 1) * per_page
    items = query.offset(offset).limit(per_page).all()
    
    meta = PaginationMeta(
        total=total,
        page=page,
        per_page=per_page,
        last_page=last_page
    )
    
    return items, meta
