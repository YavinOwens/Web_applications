// Validation results handling
function toggleAllDetails() {
    const detailButtons = document.querySelectorAll('.row-details-btn');
    const isAnyHidden = Array.from(detailButtons).some(btn => !btn.classList.contains('active'));
    
    detailButtons.forEach(btn => {
        const rowId = btn.dataset.rowId;
        if (isAnyHidden) {
            showRowDetails(rowId);
            btn.classList.add('active');
        } else {
            hideRowDetails(rowId);
            btn.classList.remove('active');
        }
    });
}

function showRowDetails(rowId) {
    const detailsRow = document.getElementById(`details-${rowId}`);
    const button = document.querySelector(`[data-row-id="${rowId}"]`);
    
    if (detailsRow) {
        detailsRow.style.display = 'table-row';
        button.classList.add('active');
        button.innerHTML = '<i class="fas fa-chevron-up"></i> Hide Details';
    }
}

function hideRowDetails(rowId) {
    const detailsRow = document.getElementById(`details-${rowId}`);
    const button = document.querySelector(`[data-row-id="${rowId}"]`);
    
    if (detailsRow) {
        detailsRow.style.display = 'none';
        button.classList.remove('active');
        button.innerHTML = '<i class="fas fa-chevron-down"></i> Show Details';
    }
}

function toggleRowDetails(rowId) {
    const detailsRow = document.getElementById(`details-${rowId}`);
    if (detailsRow.style.display === 'none') {
        showRowDetails(rowId);
    } else {
        hideRowDetails(rowId);
    }
}

function exportFailedRows(validationId) {
    window.location.href = `/export-failed-rows/${validationId}/`;
}

// Initialize tooltips and other UI elements
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
    
    // Initialize all detail rows as hidden
    const detailRows = document.querySelectorAll('.details-row');
    detailRows.forEach(row => row.style.display = 'none');
}); 