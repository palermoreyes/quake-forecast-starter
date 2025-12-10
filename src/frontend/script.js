// src/frontend/script.js

// ============================================================================
// CONFIGURACIÓN Y CONSTANTES
// ============================================================================

const CONFIG = {
    INITIAL_VIEW: {
        lat: -9.19,
        lon: -75.01,
        zoom: window.innerWidth < 768 ? 5.5 : 6
    },
    OPERATIVE_THRESHOLD: 0.12,
    MAX_ZONES: 25,
    RISK_LEVELS: {
        CRITICAL: { min: 0.30, color: '#ff0000', label: 'Crítico' },
        VERY_HIGH: { min: 0.20, color: '#ff4d4d', label: 'Muy alto' },
        HIGH: { min: 0.12, color: '#ff9f1c', label: 'Alto operativo' },
        MODERATE: { min: 0.05, color: '#ffeb3b', label: 'Leve' },
        LOW: { min: 0, color: 'transparent', label: 'Bajo' }
    },
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 2000
};

// ============================================================================
// ESTADO GLOBAL
// ============================================================================

let globalState = {
    allZones: [],
    filteredZones: [],
    mapMarkers: L.layerGroup(),
    lastEventMarker: null,
    isLoading: false,
    footerExpanded: false
};

// ============================================================================
// UTILIDADES
// ============================================================================

function sanitizeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDatePE(isoString) {
    if (!isoString) return '--';
    try {
        return new Date(isoString).toLocaleString('es-PE', {
            timeZone: 'America/Lima',
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });
    } catch (error) {
        console.error('Error formateando fecha:', error);
        return '--';
    }
}

function formatDateShort(isoString) {
    if (!isoString) return '--';
    try {
        return new Date(isoString).toLocaleDateString('es-PE', {
            timeZone: 'America/Lima',
            day: 'numeric',
            month: 'short'
        });
    } catch (error) {
        console.error('Error formateando fecha corta:', error);
        return '--';
    }
}

function getColor(prob) {
    for (const [key, level] of Object.entries(CONFIG.RISK_LEVELS)) {
        if (prob >= level.min) return level.color;
    }
    return CONFIG.RISK_LEVELS.LOW.color;
}

function getRiskLevel(prob) {
    for (const [key, level] of Object.entries(CONFIG.RISK_LEVELS)) {
        if (prob >= level.min) return level.label;
    }
    return CONFIG.RISK_LEVELS.LOW.label;
}

function showError(message, duration = 5000) {
    const toast = document.getElementById('error-toast');
    const messageEl = document.getElementById('error-message');
    
    if (toast && messageEl) {
        messageEl.textContent = message;
        toast.style.display = 'flex';
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        
        if (duration > 0) {
            setTimeout(() => {
                toast.style.display = 'none';
            }, duration);
        }
    }
}

function showSuccess(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'success-toast';
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'polite');
    toast.innerHTML = `
        <div class="success-icon" aria-hidden="true">✓</div>
        <div>${sanitizeHTML(message)}</div>
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function closeErrorToast() {
    const toast = document.getElementById('error-toast');
    if (toast) toast.style.display = 'none';
}

function showLoading(show = true) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'block' : 'none';
        overlay.setAttribute('aria-busy', show.toString());
    }
    globalState.isLoading = show;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

async function fetchWithRetry(url, options = {}, retries = CONFIG.RETRY_ATTEMPTS) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            if (i === retries - 1) throw error;
            console.warn(`Reintento ${i + 1}/${retries} para ${url}`);
            await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
        }
    }
}

// ============================================================================
// FOOTER COLAPSABLE
// ============================================================================

function toggleFooterInfo() {
    const content = document.getElementById('info-content');
    const chevron = document.getElementById('chevron-icon');
    const btn = document.getElementById('info-toggle-btn');
    
    if (!content || !chevron || !btn) return;
    
    globalState.footerExpanded = !globalState.footerExpanded;
    
    if (globalState.footerExpanded) {
        content.style.maxHeight = content.scrollHeight + 'px';
        chevron.textContent = '▲';
        btn.setAttribute('aria-expanded', 'true');
    } else {
        content.style.maxHeight = '0';
        chevron.textContent = '▼';
        btn.setAttribute('aria-expanded', 'false');
    }
}

// ============================================================================
// INICIALIZACIÓN DEL MAPA
// ============================================================================

const map = L.map('map', {
    zoomControl: false,
    zoomSnap: 0.5,
    zoomDelta: 0.5
}).setView([CONFIG.INITIAL_VIEW.lat, CONFIG.INITIAL_VIEW.lon], CONFIG.INITIAL_VIEW.zoom);

L.control.zoom({ position: 'topleft' }).addTo(map);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap, © CARTO',
    subdomains: 'abcd',
    maxZoom: 19
}).addTo(map);

globalState.mapMarkers.addTo(map);

const HomeControl = L.Control.extend({
    options: { position: 'topleft' },
    onAdd: function () {
        const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom');
        container.innerHTML = '🌎';
        container.title = 'Restaurar Vista Nacional';
        container.setAttribute('role', 'button');
        container.setAttribute('aria-label', 'Restaurar vista al mapa completo de Perú');
        container.tabIndex = 0;
        
        const resetView = () => {
            const zoom = window.innerWidth < 768 ? 5.5 : 6;
            map.flyTo([CONFIG.INITIAL_VIEW.lat, CONFIG.INITIAL_VIEW.lon], zoom, { duration: 1.5 });
        };
        
        container.onclick = resetView;
        container.onkeypress = (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                resetView();
            }
        };
        
        return container;
    }
});
map.addControl(new HomeControl());

// ============================================================================
// NAVEGACIÓN Y UI
// ============================================================================

function switchTab(tab) {
    const mapCont = document.getElementById('map-view-container');
    const listCont = document.getElementById('sidebar');
    const btnMap = document.getElementById('btn-map');
    const btnList = document.getElementById('btn-list');

    if (tab === 'map') {
        mapCont?.classList.add('active');
        listCont?.classList.remove('active');
        btnMap?.classList.add('active');
        btnList?.classList.remove('active');
        btnMap?.setAttribute('aria-current', 'true');
        btnList?.setAttribute('aria-current', 'false');
        setTimeout(() => map.invalidateSize(), 100);
    } else {
        mapCont?.classList.remove('active');
        listCont?.classList.add('active');
        btnList?.classList.add('active');
        btnMap?.classList.remove('active');
        btnList?.setAttribute('aria-current', 'true');
        btnMap?.setAttribute('aria-current', 'false');
    }
}

function initModeTabs() {
    const tabs = document.querySelectorAll('.mode-tab');
    const citizenPanel = document.getElementById('citizen-panel');
    const scientificPanel = document.getElementById('scientific-panel');

    if (!tabs.length || !citizenPanel || !scientificPanel) return;

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => {
                t.classList.remove('active');
                t.setAttribute('aria-selected', 'false');
            });
            tab.classList.add('active');
            tab.setAttribute('aria-selected', 'true');

            const mode = tab.dataset.mode;
            if (mode === 'scientific') {
                citizenPanel.classList.add('hidden');
                scientificPanel.classList.remove('hidden');
            } else {
                citizenPanel.classList.remove('hidden');
                scientificPanel.classList.add('hidden');
            }
        });
        
        // Soporte de teclado
        tab.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                tab.click();
            }
        });
    });
}

// ============================================================================
// RENDERIZADO DE ZONAS
// ============================================================================

function renderZonesList(zones) {
    const list = document.getElementById('topk-list');
    if (!list) return;

    if (!zones || zones.length === 0) {
        const searchTerm = document.getElementById('search-input')?.value || '';
        const hasFilters = searchTerm || (document.getElementById('filter-risk')?.value !== 'all');
        
        list.innerHTML = `
            <div class="empty-state" role="status">
                <div class="empty-state-icon" aria-hidden="true">🔍</div>
                <p style="font-weight: 600; margin-bottom: 8px;">
                    ${hasFilters ? 'No hay zonas que coincidan' : 'No hay datos disponibles'}
                </p>
                <p style="font-size: 0.85rem; color: var(--text-muted);">
                    ${hasFilters ? 'Intenta ajustar los filtros de búsqueda' : 'Cargando información...'}
                </p>
                ${hasFilters ? '<button onclick="clearFilters()" class="btn btn-sm btn-outline-primary mt-2">Limpiar filtros</button>' : ''}
            </div>
        `;
        return;
    }

    list.innerHTML = '';
    
    zones.forEach((item, index) => {
        const probPct = (item.prob * 100).toFixed(1);
        const rank = index + 1;
        const riskColor = getColor(item.prob);
        const riskLevel = getRiskLevel(item.prob);
        const safePlace = sanitizeHTML(item.place || 'Ubicación desconocida');

        const div = document.createElement('div');
        div.className = 'pred-item';
        div.setAttribute('role', 'button');
        div.setAttribute('tabindex', '0');
        div.setAttribute('aria-label', `Zona ${rank}: ${safePlace}, Nivel de riesgo ${riskLevel}, ${probPct}%`);
        
        div.innerHTML = `
            <div class="rank-num"
                 style="color:${riskColor}; text-shadow:0 0 10px ${riskColor}40;"
                 aria-hidden="true">
                #${rank}
            </div>
            <div class="location-info">
                <div class="location-name" style="font-weight:bold; color:#fff; font-size:0.9rem; margin-bottom:4px; display: flex; align-items: center; gap: 4px;">
                    <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${safePlace}</span>
                    <a href="https://www.google.com/maps/search/?api=1&query=${item.lat},${item.lon}"
                       target="_blank"
                       rel="noopener noreferrer"
                       style="text-decoration:none; flex-shrink: 0;"
                       aria-label="Ver en Google Maps">
                        📍
                    </a>
                </div>
                <div class="coords" style="color:#8b949e; font-size:0.75rem; margin-bottom: 4px; font-family: 'Courier New', monospace;">
                    ${item.lat.toFixed(2)}°, ${item.lon.toFixed(2)}°
                </div>
                <span class="mag-badge">
                    Mag Est: ≥ ${item.mag_pred}
                </span>
            </div>
            <div class="prob-container">
                <div class="prob-val" style="color:${riskColor}">
                    ${probPct}%
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill"
                         style="width:${Math.min(probPct, 100)}%; background-color:${riskColor}"
                         role="progressbar"
                         aria-valuenow="${probPct}"
                         aria-valuemin="0"
                         aria-valuemax="100"
                         aria-label="Nivel de riesgo ${probPct} por ciento">
                    </div>
                </div>
            </div>
        `;

        const handleClick = (e) => {
            if (e.target.tagName === 'A') return;
            if (window.innerWidth < 768) switchTab('map');
            map.flyTo([item.lat, item.lon], 10, { duration: 1 });
        };

        div.onclick = handleClick;
        div.onkeypress = (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleClick(e);
            }
        };

        list.appendChild(div);
    });
}

function renderMapMarkers(zones) {
    globalState.mapMarkers.clearLayers();

    if (!zones || zones.length === 0) return;

    zones.forEach(zone => {
        const color = getColor(zone.prob);
        if (color === 'transparent') return;

        const marker = L.circleMarker([zone.lat, zone.lon], {
            radius: 7,
            fillColor: color,
            color: '#000',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.9
        });

        const probPct = (zone.prob * 100).toFixed(1);
        const safePlace = sanitizeHTML(zone.place || 'Ubicación desconocida');
        
        const popup = `
            <div class="popup-dark">
                <div class="popup-header" style="color:${color}">RIESGO: ${probPct}%</div>
                <div class="popup-row">
                    <span class="popup-icon" aria-hidden="true">#</span> Celda: <strong>${sanitizeHTML(zone.cell_id)}</strong>
                </div>
                <div class="popup-row">
                    <span class="popup-icon" aria-hidden="true">📍</span> ${safePlace}
                </div>
                <div style="margin-top:8px;">
                    <a href="https://www.google.com/maps/search/?api=1&query=${zone.lat},${zone.lon}"
                       target="_blank"
                       rel="noopener noreferrer"
                       style="font-size:0.75rem;text-decoration:none;color:#58a6ff;border:1px solid #58a6ff;padding:2px 6px;border-radius:4px;">
                        Ver en Mapa 🌎
                    </a>
                </div>
            </div>
        `;

        marker.bindPopup(popup);
        globalState.mapMarkers.addLayer(marker);
    });
}

// ============================================================================
// FILTROS Y BÚSQUEDA
// ============================================================================

function applyFilters() {
    const searchTerm = document.getElementById('search-input')?.value.toLowerCase() || '';
    const riskFilter = document.getElementById('filter-risk')?.value || 'all';

    let filtered = [...globalState.allZones];

    if (searchTerm) {
        filtered = filtered.filter(zone => {
            const place = (zone.place || '').toLowerCase();
            const coords = `${zone.lat.toFixed(2)},${zone.lon.toFixed(2)}`;
            return place.includes(searchTerm) || coords.includes(searchTerm);
        });
    }

    if (riskFilter !== 'all') {
        filtered = filtered.filter(zone => {
            const prob = zone.prob;
            switch (riskFilter) {
                case 'critical': return prob >= 0.30;
                case 'very-high': return prob >= 0.20 && prob < 0.30;
                case 'high': return prob >= 0.12 && prob < 0.20;
                case 'moderate': return prob >= 0.05 && prob < 0.12;
                default: return true;
            }
        });
    }

    globalState.filteredZones = filtered;
    renderZonesList(filtered);
    
    const countEl = document.getElementById('zones-count');
    if (countEl) {
        const count = filtered.length;
        countEl.textContent = `${count} zona${count !== 1 ? 's' : ''} detectada${count !== 1 ? 's' : ''}`;
        countEl.setAttribute('aria-live', 'polite');
    }
}

function clearFilters() {
    const searchInput = document.getElementById('search-input');
    const filterSelect = document.getElementById('filter-risk');
    
    if (searchInput) searchInput.value = '';
    if (filterSelect) filterSelect.value = 'all';
    
    applyFilters();
    showSuccess('Filtros limpiados');
}

function initFilters() {
    const searchInput = document.getElementById('search-input');
    const filterSelect = document.getElementById('filter-risk');

    if (searchInput) {
        searchInput.addEventListener('input', debounce(applyFilters, 300));
    }

    if (filterSelect) {
        filterSelect.addEventListener('change', applyFilters);
    }
}

// ============================================================================
// CARGA DE DATOS
// ============================================================================

async function loadForecastData() {
    try {
        const data = await fetchWithRetry('/api/forecast/latest');
        
        if (!data || !data.features) {
            throw new Error('Formato de datos inválido');
        }

        const zones = data.features
            .map(feature => {
                const lat = feature.geometry.coordinates[1];
                const lon = feature.geometry.coordinates[0];

                const placeRef =
                    feature.properties.place_ref
                    || feature.properties.place
                    || `${lat.toFixed(2)}°, ${lon.toFixed(2)}°`;

                return {
                    lat,
                    lon,
                    prob: feature.properties.prob,
                    cell_id: feature.properties.cell_id,
                    place: placeRef,
                    mag_pred: feature.properties.mag_pred || '4.0+'
                };
            })
            .filter(zone => zone.prob >= CONFIG.OPERATIVE_THRESHOLD)
            .sort((a, b) => b.prob - a.prob);

        // 👇 nos quedamos SOLO con las top-K zonas
        const topZones = zones.slice(0, CONFIG.MAX_ZONES);

        globalState.allZones = topZones;
        globalState.filteredZones = topZones;

        // Mapa y lista trabajan solo con esas 25
        renderMapMarkers(topZones);
        applyFilters();

return topZones;

    } catch (error) {
        console.error('Error cargando pronóstico:', error);
        showError('Error al cargar los datos de pronóstico. Reintentando...');
        throw error;
    }
}

async function loadLastEvent() {
    try {
        const event = await fetchWithRetry('/api/forecast/last-event');
        
        if (!event) {
            const cont = document.getElementById('lq-content');
            if (cont) cont.innerHTML = '<span class="lq-loading">Sin datos disponibles</span>';
            return;
        }

        const cont = document.getElementById('lq-content');
        if (cont) {
            const safeMag = parseFloat(event.magnitude).toFixed(1);
            const safePlace = sanitizeHTML(event.place || 'Ubicación desconocida');
            const safeTime = formatDatePE(event.event_time_utc);
            
            cont.innerHTML = `
                <div class="d-flex align-items-center justify-content-center" style="min-width: 60px;">
                    <div class="lq-mag">${safeMag}</div>
                    <div class="lq-unit ms-1" style="font-size:0.65rem; color:#8b949e; line-height:1.1;">MAG<br>MW</div>
                </div>
                
                <div class="lq-details ms-2" style="flex-grow:1; text-align:right;">
                    <div class="lq-row" style="justify-content:flex-end;">
                        <span class="lq-icon" aria-hidden="true">📍</span> ${safePlace}
                    </div>
                    <div class="lq-row" style="justify-content:flex-end;">
                        <span class="lq-icon" aria-hidden="true">🕒</span> ${safeTime}
                    </div>
                    <div style="margin-top:2px;">
                        <a href="https://ultimosismo.igp.gob.pe/"
                           target="_blank"
                           rel="noopener noreferrer"
                           class="btn-igp-mini"
                           style="color:#58a6ff; font-size:0.65rem; text-decoration:none;">
                            🌐 Ver en IGP ↗
                        </a>
                    </div>
                </div>
            `;
        }

        if (globalState.lastEventMarker) {
            map.removeLayer(globalState.lastEventMarker);
        }

        const lastIcon = L.divIcon({
            className: 'custom-div-icon',
            html: `<div style="background:rgba(248,81,73,0.3); width:40px; height:40px; border-radius:50%; border:2px solid #f85149; animation:pulse-red 2s infinite;"></div>`,
            iconSize: [40, 40],
            iconAnchor: [20, 20]
        });

        const safePlace = sanitizeHTML(event.place || 'Ubicación desconocida');
        const popup = `
            <div class="popup-dark">
                <div class="popup-header" style="color:#f85149">🔴 Último Sismo</div>
                <div class="popup-row">
                    <span class="popup-icon" aria-hidden="true">📍</span> ${safePlace}
                </div>
                <div class="popup-row">
                    <span class="popup-icon" aria-hidden="true">⚡</span> Mag: <strong>${parseFloat(event.magnitude).toFixed(1)}</strong>
                </div>
                <div class="popup-row">
                    <span class="popup-icon" aria-hidden="true">🕒</span> ${formatDatePE(event.event_time_utc).split(',')[1] || '--'}
                </div>
                <div class="popup-row">
                    <span class="popup-icon" aria-hidden="true">📉</span> Prof: <strong>${parseFloat(event.depth_km).toFixed(0)} km</strong>
                </div>
            </div>
        `;

        globalState.lastEventMarker = L.marker([event.lat, event.lon], { icon: lastIcon })
            .addTo(map)
            .bindPopup(popup);

    } catch (error) {
        console.error('Error cargando último evento:', error);
        const cont = document.getElementById('lq-content');
        if (cont) cont.innerHTML = '<span class="lq-loading" style="color:#f85149">Error al cargar</span>';
    }
}

async function loadMetadata() {
    try {
        const data = await fetchWithRetry('/api/forecast/topk');
        
        if (!data || !data.topk || !data.topk.length) {
            throw new Error('No hay datos de metadatos disponibles');
        }

        const genDate = document.getElementById('generated-date');
        const inputDate = document.getElementById('input-date');
        const validityRange = document.getElementById('validity-range');
        const modelBadge = document.getElementById('model-badge');

        if (genDate) genDate.textContent = formatDatePE(data.generated_at);
        if (inputDate) {
            inputDate.textContent = new Date(data.input_max_time).toLocaleDateString('es-PE', {
                timeZone: 'America/Lima'
            });
        }

        if (validityRange && data.topk[0]) {
            const vStart = new Date(data.topk[0].t_pred_start);
            const vEnd = new Date(data.topk[0].t_pred_end);
            validityRange.textContent = 
                `${formatDateShort(vStart)} al ${formatDateShort(vEnd)}`;
        }

        if (modelBadge && data.model_version) {
            modelBadge.textContent = data.model_version;
        } else if (modelBadge && modelBadge.textContent === 'Cargando...') {
            modelBadge.textContent = 'LSTM v3.3.1';
        }

        const maxProb = globalState.allZones.length > 0 
            ? Math.max(...globalState.allZones.map(z => z.prob))
            : 0;
            
        updateRecommendation(maxProb);
        updateScientificPanel();

    } catch (error) {
        console.error('Error cargando metadatos:', error);
        showError('Error al cargar metadatos del sistema');
    }
}

function updateRecommendation(maxProb) {
    const recBox = document.getElementById('recommendation-box');
    const recText = document.getElementById('recommendation-text');
    
    if (!recBox || !recText) return;

    if (maxProb >= 0.25) {
        recBox.style.display = 'flex';
        recBox.classList.add('critical');
        recText.innerHTML =
            "<strong>NIVEL CRÍTICO LOCAL:</strong> Una o más zonas presentan valores de riesgo ≥ 25%. " +
            "Se recomienda revisar planes de emergencia y fuentes oficiales (INDECI / IGP).";
    } else if (maxProb >= CONFIG.OPERATIVE_THRESHOLD) {
        recBox.style.display = 'flex';
        recBox.classList.remove('critical');
        recText.innerHTML =
            "<strong>PRECAUCIÓN:</strong> El modelo detecta actividad inusual en algunas zonas. " +
            "Es recomendable verificar sus mochilas de emergencia y rutas de evacuación.";
    } else {
        recBox.style.display = 'none';
        recBox.classList.remove('critical');
    }
}

function updateScientificPanel() {
    const zones = globalState.allZones;
    
    if (!zones || zones.length === 0) {
        const setText = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };
        
        setText('sci-total', 'Sin datos');
        setText('sci-visible', 'Sin datos');
        setText('sci-maxprob', '--');
        setText('sci-avgprob', '--');
        setText('sci-above', '0 zonas');
        setText('sci-lat-range', '--');
        setText('sci-lon-range', '--');
        return;
    }

    const probs = zones.map(z => z.prob);
    const lats = zones.map(z => z.lat);
    const lons = zones.map(z => z.lon);

    const maxProb = Math.max(...probs);
    const avgProb = probs.reduce((a, b) => a + b, 0) / probs.length;
    const countAbove = probs.filter(p => p >= CONFIG.OPERATIVE_THRESHOLD).length;
    const pctAbove = (countAbove / probs.length) * 100;

    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    const modelBadge = document.getElementById('model-badge');
    const modelName = modelBadge ? modelBadge.textContent : 'LSTM v3.3.1';
    setText('sci-model', `${modelName} – Bi-LSTM 30 días`);
    setText('sci-total', `${zones.length} celdas`);
    setText('sci-visible', `${zones.length} celdas`);
    setText('sci-maxprob', `${(maxProb * 100).toFixed(1)}%`);
    setText('sci-avgprob', `${(avgProb * 100).toFixed(1)}%`);
    setText('sci-threshold', `${(CONFIG.OPERATIVE_THRESHOLD * 100).toFixed(1)}%`);
    setText('sci-above', `${countAbove} zonas (${pctAbove.toFixed(0)}%)`);

    const minLat = Math.min(...lats).toFixed(1);
    const maxLat = Math.max(...lats).toFixed(1);
    const minLon = Math.min(...lons).toFixed(1);
    const maxLon = Math.max(...lons).toFixed(1);

    setText('sci-lat-range', `${minLat}° a ${maxLat}°`);
    setText('sci-lon-range', `${minLon}° a ${maxLon}°`);
}

// ============================================================================
// LEYENDA DEL MAPA
// ============================================================================

function initLegend() {
    const legend = L.control({ position: 'bottomleft' });

    legend.onAdd = function () {
        const div = L.DomUtil.create('div', 'info legend');
        div.setAttribute('role', 'region');
        div.setAttribute('aria-label', 'Leyenda del mapa de riesgo sísmico');
        
        const grades = [
            { label: 'Crítico (≥30%)', color: CONFIG.RISK_LEVELS.CRITICAL.color },
            { label: 'Muy alto (20–30%)', color: CONFIG.RISK_LEVELS.VERY_HIGH.color },
            { label: 'Alto operativo (12–20%)', color: CONFIG.RISK_LEVELS.HIGH.color },
            { label: 'Leve (5–12%)', color: CONFIG.RISK_LEVELS.MODERATE.color }
        ];

        let html = '<h6>Nivel de riesgo</h6>';
        grades.forEach(i => {
            html += `
                <div class="legend-item">
                    <i class="legend-icon" style="background:${i.color}" aria-hidden="true"></i>
                    <span>${i.label}</span>
                </div>
            `;
        });
        html += `
            <div style="margin-top:8px; font-size:0.65rem; color:#8b949e">
                Score relativo de riesgo de sismo ≥ M4.0 (ventana 7 días)
            </div>
        `;
        div.innerHTML = html;
        return div;
    };

    legend.addTo(map);
}

// ============================================================================
// FUNCIÓN DE REFRESCO
// ============================================================================

async function refreshData() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.classList.add('spinning');
        refreshBtn.setAttribute('aria-busy', 'true');
    }

    showLoading(true);

    try {
        await loadForecastData();
        await loadLastEvent();
        await loadMetadata();
        
        showLoading(false);
        showSuccess('✓ Datos actualizados correctamente');
    } catch (error) {
        showLoading(false);
        showError('Error al actualizar los datos. Por favor, intente nuevamente.', 10000);
    } finally {
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.classList.remove('spinning');
            refreshBtn.setAttribute('aria-busy', 'false');
        }
    }
}

// ============================================================================
// RESPONSIVE Y REDIMENSIONAMIENTO
// ============================================================================

let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        map.invalidateSize();
    }, 150);
});

// ============================================================================
// INICIALIZACIÓN
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Mostrar modal de bienvenida solo la primera vez
        const welcomeSeen = localStorage.getItem('quakeForecastWelcomeSeen');
        
        if (!welcomeSeen) {
            const modalEl = document.getElementById('welcomeModal');
            if (modalEl) {
                const welcomeModal = new bootstrap.Modal(modalEl, {
                    backdrop: 'static',
                    keyboard: true
                });
                welcomeModal.show();
                
                // Guardar en localStorage cuando se cierre
                modalEl.addEventListener('hidden.bs.modal', function() {
                    localStorage.setItem('quakeForecastWelcomeSeen', 'true');
                });
            }
        }
        
        showLoading(true);
        
        // Inicializar UI
        initModeTabs();
        initFilters();
        initLegend();
        
        // Cargar datos EN ORDEN
        await loadForecastData();
        await loadLastEvent();
        await loadMetadata();
        
        showLoading(false);
        
        // Estado inicial en móvil - mostrar lista primero
        if (window.innerWidth < 768) {
            switchTab('list');
        }
        
    } catch (error) {
        console.error('Error en inicialización:', error);
        showLoading(false);
        showError('Error al inicializar la aplicación. Por favor, recargue la página.', 0);
    }
});

// ============================================================================
// EXPORTAR FUNCIONES GLOBALES PARA USO EN HTML
// ============================================================================

window.switchTab = switchTab;
window.refreshData = refreshData;
window.closeErrorToast = closeErrorToast;
window.clearFilters = clearFilters;
window.toggleFooterInfo = toggleFooterInfo;