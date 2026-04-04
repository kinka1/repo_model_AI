/**
 * Reusable Pagination Component (Vanilla JS)
 * Membantu merender kontrol navigasi halaman dan mengelola callback perubahan halaman.
 */
class PaginationComponent {
    /**
     * @param {string} containerId - ID elemen HTML tempat pagination akan dirender.
     * @param {Function} onPageChange - Callback yang dipanggil saat halaman berubah (menerima param 'page').
     */
    constructor(containerId, onPageChange) {
        this.container = document.getElementById(containerId);
        this.onPageChange = onPageChange;
        
        // Daftarkan ke window agar bisa dipanggil dari inline onclick
        window._paginationInstances = window._paginationInstances || {};
        window._paginationInstances[containerId] = this;
    }

    /**
     * Render elemen pagination berdasarkan metadata dari API.
     * @param {Object} meta - Objek metadata (page, last_page, per_page, total).
     */
    render(meta) {
        if (!this.container) return;
        
        const { page, last_page, total } = meta;
        const containerId = this.container.id;

        // Jika data kosong atau hanya 1 halaman, kosongkan container
        if (!total || last_page <= 1) {
            this.container.innerHTML = '';
            return;
        }

        let buttonsHtml = '';
        
        // Tombol Previous
        buttonsHtml += `
            <button class="pag-btn" ${page === 1 ? 'disabled' : ''} 
                onclick="window._paginationInstances['${containerId}'].goToPage(${page - 1})">
                &laquo; Prev
            </button>
        `;

        // Logika range halaman (menampilkan max 5 nomor halaman di sekitar current page)
        let startPage = Math.max(1, page - 2);
        let endPage = Math.min(last_page, startPage + 4);
        
        if (endPage - startPage < 4) {
            startPage = Math.max(1, endPage - 4);
        }

        if (startPage > 1) {
            buttonsHtml += `<button class="pag-btn" onclick="window._paginationInstances['${containerId}'].goToPage(1)">1</button>`;
            if (startPage > 2) buttonsHtml += `<span class="pag-dots">...</span>`;
        }

        for (let i = startPage; i <= endPage; i++) {
            buttonsHtml += `
                <button class="pag-btn ${i === page ? 'active' : ''}" 
                    ${i === page ? 'disabled' : ''} 
                    onclick="window._paginationInstances['${containerId}'].goToPage(${i})">
                    ${i}
                </button>
            `;
        }

        if (endPage < last_page) {
            if (endPage < last_page - 1) buttonsHtml += `<span class="pag-dots">...</span>`;
            buttonsHtml += `<button class="pag-btn" onclick="window._paginationInstances['${containerId}'].goToPage(${last_page})">${last_page}</button>`;
        }

        // Tombol Next
        buttonsHtml += `
            <button class="pag-btn" ${page === last_page ? 'disabled' : ''} 
                onclick="window._paginationInstances['${containerId}'].goToPage(${page + 1})">
                Next &raquo;
            </button>
        `;

        this.container.innerHTML = `
            <div class="pagination-container">
                <style>
                    .pagination-container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        gap: 12px;
                        margin-top: 2.5rem;
                        padding: 1rem;
                        background: rgba(255,255,255,0.4);
                        border-radius: 16px;
                    }
                    .pagination-buttons {
                        display: flex;
                        align-items: center;
                        gap: 6px;
                        flex-wrap: wrap;
                        justify-content: center;
                    }
                    .pag-btn {
                        background: white;
                        border: 1px solid #e2e8f0;
                        color: #1e293b;
                        padding: 8px 14px;
                        font-size: 0.9rem;
                        font-weight: 500;
                        border-radius: 8px;
                        cursor: pointer;
                        transition: all 0.2s;
                    }
                    .pag-btn:hover:not(:disabled) {
                        border-color: #2563eb;
                        color: #2563eb;
                        background: #f0f9ff;
                    }
                    .pag-btn.active {
                        background: #2563eb;
                        color: white;
                        border-color: #2563eb;
                        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
                    }
                    .pag-btn:disabled {
                        opacity: 0.5;
                        cursor: not-allowed;
                    }
                    .pag-dots {
                        color: #64748b;
                        padding: 0 4px;
                    }
                    .pagination-info {
                        font-size: 0.8rem;
                        color: #64748b;
                        font-weight: 400;
                    }
                </style>
                <div class="pagination-buttons">
                    ${buttonsHtml}
                </div>
                <div class="pagination-info">
                    Halaman <strong>${page}</strong> dari <strong>${last_page}</strong> (Total <strong>${total}</strong> data)
                </div>
            </div>
        `;
    }

    /**
     * Memanggil callback untuk berpindah halaman.
     * @param {number} pageNum 
     */
    goToPage(pageNum) {
        if (this.onPageChange) {
            this.onPageChange(pageNum);
        }
    }
}
