// Shared helpers
function postJSON(url, body) {
    return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    }).then(r => r.json().then(data => ({ ok: r.ok, status: r.status, data })));
}

function putJSON(url, body) {
    return fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    }).then(r => r.json().then(data => ({ ok: r.ok, status: r.status, data })));
}

function deleteReq(url) {
    return fetch(url, { method: 'DELETE' })
        .then(r => r.json().then(data => ({ ok: r.ok, status: r.status, data })));
}

// ============================================================
// Dashboard page
// ============================================================
let leafletMap = null;
let markers = [];
let userMarker = null;
let businessMarkers = [];
let selectedBusinessIdx = null;

function initDashboardMap() {
    const lat = window.USER_LAT || 27.9483;
    const lng = window.USER_LNG || -82.4648;

    leafletMap = L.map('map').setView([lat, lng], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(leafletMap);

    // Add user marker (blue circle) and keep it persistent
    userMarker = blueMarker(lat, lng).addTo(leafletMap).bindPopup('You are here');
    markers.push(userMarker);
}

function clearBusinessMarkers() {
    businessMarkers.forEach(m => leafletMap.removeLayer(m));
    businessMarkers = [];
}

function redMarker(lat, lng) {
    return L.marker([lat, lng], {
        icon: L.divIcon({
            className: 'biz-marker',
            html: '<div style="background:#d32323;width:16px;height:16px;border-radius:50%;border:2.5px solid #fff;box-shadow:0 2px 6px rgba(211,35,35,0.45),0 0 0 1px rgba(0,0,0,0.08);"></div>',
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        })
    });
}

function blueMarker(lat, lng) {
    return L.marker([lat, lng], {
        icon: L.divIcon({
            className: 'user-marker',
            html: '<div style="position:relative;width:22px;height:22px;"><div style="position:absolute;inset:0;background:#2563eb;border-radius:50%;border:3px solid #fff;box-shadow:0 2px 8px rgba(37,99,235,0.45),0 0 0 1px rgba(0,0,0,0.08);"></div><div style="position:absolute;inset:-6px;border:2px solid rgba(37,99,235,0.35);border-radius:50%;animation:userPulse 2s ease-out infinite;"></div></div>',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
        })
    });
}

function renderResults(data) {
    // Clear business markers only (user marker persists)
    clearBusinessMarkers();

    const body = document.getElementById('results-body');
    const header = document.getElementById('results-header');
    body.innerHTML = '';

    // Build header based on options
    let headerHTML = '<th>Name</th><th>City</th><th>Reviews</th><th>Stars</th>';
    if (data.trip_duration) headerHTML += '<th>Trip</th>';
    if (data.detail) headerHTML += '<th>Categories</th>';
    header.innerHTML = headerHTML;

    const bounds = [[data.user_lat, data.user_lng]];
    const results = data.results || [];

    results.forEach((r, idx) => {
        // Table row
        const tr = document.createElement('tr');
        tr.setAttribute('data-business-idx', idx);
        let rowHTML = `<td>${escapeHtml(r.name || '')}</td>` +
                      `<td>${escapeHtml(r.city || '')}</td>` +
                      `<td>${r.review_count ?? ''}</td>` +
                      `<td>${r.stars ?? ''}</td>`;
        if (data.trip_duration) rowHTML += `<td>${escapeHtml(r.trip_duration || '')}</td>`;
        if (data.detail) rowHTML += `<td>${escapeHtml(r.categories || '')}</td>`;
        tr.innerHTML = rowHTML;
        body.appendChild(tr);

        // Marker (red circles for businesses)
        if (r.latitude != null && r.longitude != null) {
            const m = redMarker(r.latitude, r.longitude).addTo(leafletMap);
            const popup = `<strong>${escapeHtml(r.name || '')}</strong> &mdash; ${r.stars ?? '?'}&#9733;`;
            m.bindPopup(popup);
            businessMarkers.push({ marker: m, row: tr });
            bounds.push([r.latitude, r.longitude]);

            // Add hover/click highlighting
            tr.addEventListener('mouseenter', () => {
                if (selectedBusinessIdx === null) highlightBusiness(idx);
            });
            tr.addEventListener('mouseleave', () => {
                if (selectedBusinessIdx === null) unhighlightBusiness(idx);
            });
            tr.addEventListener('click', (ev) => {
                ev.stopPropagation();
                selectBusiness(idx);
            });
        }
    });

    if (bounds.length > 1) {
        leafletMap.fitBounds(bounds, { padding: [30, 30] });
    }
}

function selectBusiness(idx) {
    const prev = selectedBusinessIdx;
    selectedBusinessIdx = idx;

    const rows = document.querySelectorAll('#results-body tr');
    rows.forEach((r, i) => {
        if (i === idx) {
            r.classList.add('highlighted-row');
            r.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            r.classList.remove('highlighted-row');
        }
    });

    if (prev !== null && businessMarkers[prev] && businessMarkers[prev].marker) {
        businessMarkers[prev].marker.closePopup();
    }
    if (businessMarkers[idx] && businessMarkers[idx].marker) {
        businessMarkers[idx].marker.openPopup();
    }
}

function clearSelection() {
    if (selectedBusinessIdx !== null) {
        const rows = document.querySelectorAll('#results-body tr');
        rows.forEach(r => r.classList.remove('highlighted-row'));
        if (businessMarkers[selectedBusinessIdx] && businessMarkers[selectedBusinessIdx].marker) {
            businessMarkers[selectedBusinessIdx].marker.closePopup();
        }
        selectedBusinessIdx = null;
    }
}

function highlightBusiness(idx) {
    const rows = document.querySelectorAll('#results-body tr');
    rows.forEach((r, i) => {
        if (i === idx) {
            r.classList.add('highlighted-row');
            r.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            r.classList.remove('highlighted-row');
        }
    });
    if (businessMarkers[idx] && businessMarkers[idx].marker) {
        businessMarkers[idx].marker.openPopup();
    }
}

function unhighlightBusiness(idx) {
    const rows = document.querySelectorAll('#results-body tr');
    rows.forEach(r => r.classList.remove('highlighted-row'));
    if (businessMarkers[idx] && businessMarkers[idx].marker) {
        businessMarkers[idx].marker.closePopup();
    }
}

function escapeHtml(s) {
    if (s == null) return '';
    return String(s)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
}

function setupLocalityToggle() {
    const sel = document.getElementById('locality_mode');
    const stateF = document.getElementById('state-fields');
    const cityF = document.getElementById('city-fields');
    const radialF = document.getElementById('radial-fields');

    function update() {
        stateF.style.display = sel.value === 'state' ? '' : 'none';
        cityF.style.display = sel.value === 'city' ? '' : 'none';
        radialF.style.display = sel.value === 'radial' ? '' : 'none';
    }
    sel.addEventListener('change', update);
    update();
}

function setupRadiusSlider() {
    const slider = document.getElementById('radius');
    const out = document.getElementById('radius-value');
    if (!slider || !out) return;
    const sync = () => { out.textContent = slider.value; };
    slider.addEventListener('input', sync);
    sync();
}

function setupNSlider() {
    const slider = document.getElementById('n');
    const out = document.getElementById('n-value');
    if (!slider || !out) return;
    const sync = () => { out.textContent = slider.value; };
    slider.addEventListener('input', sync);
    sync();
}

// ------------------------------------------------------------
// Category multi-select picker (backed by Word2Vec vocabulary)
// ------------------------------------------------------------
const categoryState = {
    vocab: [],
    selected: new Set(['Bars', 'Nightlife']),
    highlightIdx: -1,
    suggestions: [],
};

async function loadCategoryVocab() {
    try {
        const r = await fetch('/categories');
        const data = await r.json();
        categoryState.vocab = Array.isArray(data.categories) ? data.categories : [];
    } catch (e) {
        categoryState.vocab = [];
    }
}

function renderCategoryChips() {
    const host = document.getElementById('category-chips');
    if (!host) return;
    host.innerHTML = '';
    categoryState.selected.forEach(cat => {
        const chip = document.createElement('span');
        chip.className = 'category-chip';
        chip.innerHTML = `<span class="chip-label"></span><button type="button" class="chip-remove" aria-label="Remove">&times;</button>`;
        chip.querySelector('.chip-label').textContent = cat;
        chip.querySelector('.chip-remove').addEventListener('click', () => {
            categoryState.selected.delete(cat);
            renderCategoryChips();
        });
        host.appendChild(chip);
    });
}

function renderCategorySuggestions(query) {
    const list = document.getElementById('category-suggestions');
    if (!list) return;
    const q = query.trim().toLowerCase();
    if (!q) {
        list.hidden = true;
        list.innerHTML = '';
        categoryState.suggestions = [];
        categoryState.highlightIdx = -1;
        return;
    }

    const matches = [];
    for (const cat of categoryState.vocab) {
        if (categoryState.selected.has(cat)) continue;
        if (cat.toLowerCase().includes(q)) matches.push(cat);
        if (matches.length >= 50) break;
    }
    categoryState.suggestions = matches;
    categoryState.highlightIdx = matches.length ? 0 : -1;

    list.innerHTML = '';
    matches.forEach((cat, i) => {
        const li = document.createElement('li');
        li.className = 'suggestion-item' + (i === categoryState.highlightIdx ? ' highlight' : '');
        li.textContent = cat;
        li.addEventListener('mousedown', (ev) => {
            ev.preventDefault();
            selectCategory(cat);
        });
        list.appendChild(li);
    });
    list.hidden = matches.length === 0;
}

function selectCategory(cat) {
    categoryState.selected.add(cat);
    const input = document.getElementById('category-search');
    if (input) input.value = '';
    renderCategoryChips();
    renderCategorySuggestions('');
}

function setupCategoryPicker() {
    const input = document.getElementById('category-search');
    const list = document.getElementById('category-suggestions');
    if (!input || !list) return;

    renderCategoryChips();

    input.addEventListener('input', () => renderCategorySuggestions(input.value));
    input.addEventListener('focus', () => renderCategorySuggestions(input.value));
    input.addEventListener('blur', () => {
        setTimeout(() => { list.hidden = true; }, 120);
    });

    input.addEventListener('keydown', (ev) => {
        const n = categoryState.suggestions.length;
        if (ev.key === 'ArrowDown' && n) {
            ev.preventDefault();
            categoryState.highlightIdx = (categoryState.highlightIdx + 1) % n;
            renderCategorySuggestions(input.value);
        } else if (ev.key === 'ArrowUp' && n) {
            ev.preventDefault();
            categoryState.highlightIdx = (categoryState.highlightIdx - 1 + n) % n;
            renderCategorySuggestions(input.value);
        } else if (ev.key === 'Enter' && categoryState.highlightIdx >= 0) {
            ev.preventDefault();
            selectCategory(categoryState.suggestions[categoryState.highlightIdx]);
        } else if (ev.key === 'Backspace' && !input.value && categoryState.selected.size) {
            const last = Array.from(categoryState.selected).pop();
            categoryState.selected.delete(last);
            renderCategoryChips();
        } else if (ev.key === 'Escape') {
            list.hidden = true;
        }
    });
}

function setDefaultDay() {
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const today = days[new Date().getDay()];
    const sel = document.getElementById('day');
    if (sel) sel.value = today;
}

function setDefaultTime() {
    const t = document.getElementById('time');
    if (!t) return;
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    t.value = `${hh}:${mm}`;
}

async function initDashboard() {
    initDashboardMap();
    setupLocalityToggle();
    setupRadiusSlider();
    setupNSlider();
    setDefaultDay();
    setDefaultTime();
    await loadCategoryVocab();
    setupCategoryPicker();

    // Clear business selection when clicking anywhere on the page
    document.addEventListener('click', (ev) => {
        if (!ev.target.closest('#results-body')) {
            clearSelection();
        }
    });

    const form = document.getElementById('recommend-form');
    const btn = document.getElementById('submit-btn');

    form.addEventListener('submit', async (ev) => {
        ev.preventDefault();

        const localityMode = document.getElementById('locality_mode').value;
        const timeStr = document.getElementById('time').value || '21:00';
        const [hh, mm] = timeStr.split(':').map(Number);
        const timeMinutes = (hh || 0) * 60 + (mm || 0);

        const selectedCats = Array.from(categoryState.selected);
        if (!selectedCats.length) {
            alert('Please select at least one category.');
            return;
        }

        const body = {
            locality_mode: localityMode,
            categories: selectedCats,
            scope: parseInt(document.getElementById('scope').value, 10),
            day: document.getElementById('day').value,
            time_minutes: timeMinutes,
            n: parseInt(document.getElementById('n').value, 10) || 5,
            currently_open: document.getElementById('currently_open').checked,
            recommend_reviewed: document.getElementById('recommend_reviewed').checked,
            trip_duration: document.getElementById('trip_duration').checked,
            detail: document.getElementById('detail').checked,
        };

        if (localityMode === 'state') {
            body.state = document.getElementById('state').value;
        } else if (localityMode === 'city') {
            body.city = document.getElementById('city').value;
        } else if (localityMode === 'radial') {
            body.latitude = parseFloat(document.getElementById('latitude').value);
            body.longitude = parseFloat(document.getElementById('longitude').value);
            body.radius = parseFloat(document.getElementById('radius').value);
        }

        btn.disabled = true;
        const origText = btn.textContent;
        btn.textContent = 'Loading...';

        try {
            const { ok, data } = await postJSON('/recommend', body);
            if (!ok || data.error) {
                alert('Error: ' + (data.error || 'unknown'));
            } else {
                renderResults(data);
            }
        } catch (e) {
            alert('Request failed: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = origText;
        }
    });
}

// ============================================================
// Profile page
// ============================================================
function initProfile() {
    // Save location
    const saveLocBtn = document.getElementById('save-location-btn');
    if (saveLocBtn) {
        saveLocBtn.addEventListener('click', async () => {
            const latEl = document.getElementById('loc_lat');
            const lngEl = document.getElementById('loc_lng');
            const lat = parseFloat(latEl.value || latEl.placeholder);
            const lng = parseFloat(lngEl.value || lngEl.placeholder);
            if (isNaN(lat) || isNaN(lng)) {
                alert('Please provide valid latitude and longitude.');
                return;
            }
            const { ok, data } = await postJSON('/profile/edit', { latitude: lat, longitude: lng });
            if (!ok || data.error) {
                alert('Error: ' + (data.error || 'unknown'));
            } else {
                alert('Location saved.');
                location.reload();
            }
        });
    }

    // Per-review save / delete
    document.querySelectorAll('.review-card').forEach(card => {
        const rid = card.getAttribute('data-review-id');

        const saveBtn = card.querySelector('.review-save-btn');
        const delBtn = card.querySelector('.review-delete-btn');

        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                const stars = parseFloat(card.querySelector('.review-stars').value);
                const text = card.querySelector('.review-text').value;
                const { ok, data } = await putJSON(`/profile/review/${encodeURIComponent(rid)}`, { stars, text });
                if (!ok || data.error) {
                    alert('Error: ' + (data.error || 'unknown'));
                } else {
                    alert('Review updated.');
                    location.reload();
                }
            });
        }

        if (delBtn) {
            delBtn.addEventListener('click', async () => {
                if (!confirm('Delete this review?')) return;
                const { ok, data } = await deleteReq(`/profile/review/${encodeURIComponent(rid)}`);
                if (!ok || data.error) {
                    alert('Error: ' + (data.error || 'unknown'));
                } else {
                    location.reload();
                }
            });
        }
    });

    // Add review
    const addForm = document.getElementById('add-review-form');
    if (addForm) {
        addForm.addEventListener('submit', async (ev) => {
            ev.preventDefault();
            const body = {
                business_id: document.getElementById('new_business_id').value.trim(),
                stars: parseFloat(document.getElementById('new_stars').value),
                text: document.getElementById('new_text').value,
            };
            // Generate a simple review_id if backend did not already
            body.review_id = 'web-' + Math.random().toString(36).slice(2, 12) + Date.now().toString(36);
            const { ok, data } = await postJSON('/profile/review', body);
            if (!ok || data.error) {
                alert('Error: ' + (data.error || 'unknown'));
            } else {
                alert('Review added.');
                location.reload();
            }
        });
    }
}

// ============================================================
// Entry point
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('recommend-form')) {
        initDashboard();
    } else if (document.getElementById('reviews-list') || document.getElementById('location-card')) {
        initProfile();
    }
});
