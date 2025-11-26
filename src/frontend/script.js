// src/frontend/script.js

// 1. Configuración Inicial INTELIGENTE
const isMobile = window.innerWidth < 768;

const INITIAL_VIEW = { 
    lat: -9.19, 
    lon: -75.01, 
    // CAMBIO: Zoom 5.5 para móvil (punto medio perfecto)
    zoom: isMobile ? 5.5 : 6 
};

// CAMBIO: zoomSnap 0.5 permite niveles intermedios
const map = L.map('map', { 
    zoomControl: false,
    zoomSnap: 0.5, 
    zoomDelta: 0.5
}).setView([INITIAL_VIEW.lat, INITIAL_VIEW.lon], INITIAL_VIEW.zoom); 

L.control.zoom({ position: 'topleft' }).addTo(map);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap, © CARTO',
    subdomains: 'abcd',
    maxZoom: 19
}).addTo(map);

// Botón Restaurar
const HomeControl = L.Control.extend({
    options: { position: 'topleft' },
    onAdd: function(map) {
        const c = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom');
        c.innerHTML = '🌍'; 
        c.title = "Restaurar Vista Nacional";
        c.onclick = function(){ 
            const z = window.innerWidth < 768 ? 5.5 : 6;
            map.flyTo([INITIAL_VIEW.lat, INITIAL_VIEW.lon], z, { duration: 1.5 }); 
        }; 
        return c;
    }
});
map.addControl(new HomeControl());

// --- LÓGICA DE PESTAÑAS (MÓVIL) ---
function switchTab(tab) {
    const mapCont = document.getElementById('map-view-container');
    const listCont = document.getElementById('sidebar');
    const btnMap = document.getElementById('btn-map');
    const btnList = document.getElementById('btn-list');

    if (tab === 'map') {
        mapCont.classList.add('active');
        listCont.classList.remove('active');
        btnMap.classList.add('active');
        btnList.classList.remove('active');
        map.invalidateSize();
    } else {
        mapCont.classList.remove('active');
        listCont.classList.add('active');
        btnList.classList.add('active');
        btnMap.classList.remove('active');
    }
}

// Utilitarios
function formatDatePE(isoString) { if (!isoString) return "--"; return new Date(isoString).toLocaleString('es-PE', { timeZone: 'America/Lima', day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: true }); }
function getColor(prob) { return prob > 0.95 ? '#ff0000' : prob > 0.85 ? '#ff4d4d' : prob > 0.70 ? '#ff9f1c' : prob > 0.55 ? '#ffeb3b' : 'transparent'; }

// 1. Cargar Mapa
fetch('/api/forecast/latest').then(r => r.json()).then(data => {
    L.geoJson(data, {
        pointToLayer: (feature, latlng) => {
            if (feature.properties.prob < 0.55) return null;
            return L.circleMarker(latlng, { radius: 7, fillColor: getColor(feature.properties.prob), color: "#000", weight: 1, opacity: 1, fillOpacity: 0.9 });
        },
        onEachFeature: (feature, layer) => {
            if (layer) {
                const p = (feature.properties.prob * 100).toFixed(1); const lat = feature.geometry.coordinates[1].toFixed(3); const lon = feature.geometry.coordinates[0].toFixed(3); const col = getColor(feature.properties.prob);
                layer.bindPopup(`<div class="popup-dark"><div class="popup-header" style="color:${col}">RIESGO: ${p}%</div><div class="popup-row">Celda: <strong>${feature.properties.cell_id}</strong></div><div style="margin-top:8px;"><a href="https://www.google.com/maps/search/?api=1&query=${lat},${lon}" target="_blank" style="font-size:0.75rem;text-decoration:none;color:#58a6ff;border:1px solid #58a6ff;padding:2px 6px;border-radius:4px;">Ver en Mapa 🌍</a></div></div>`);
            }
        }
    }).addTo(map);
});

// 2. Widget Sismo (CON LINK IGP)
fetch('/api/forecast/last-event').then(r => r.json()).then(event => {
    if (!event) { document.getElementById('lq-content').innerHTML = 'Sin datos.'; return; }
    
    document.getElementById('lq-content').innerHTML = `
        <div class="d-flex align-items-center justify-content-center" style="min-width: 60px;">
            <div class="lq-mag">${event.magnitude.toFixed(1)}</div>
            <div class="lq-unit ms-1" style="font-size:0.65rem; color:#8b949e; line-height:1.1;">MAG<br>MW</div>
        </div>
        
        <div class="lq-details ms-2" style="flex-grow:1; text-align:right;">
            <div class="lq-row" style="justify-content:flex-end;"><span class="lq-icon">📍</span> ${event.place}</div>
            <div class="lq-row" style="justify-content:flex-end;">
                <span class="lq-icon">🕒</span> ${formatDatePE(event.event_time_utc)}
            </div>
            <div style="margin-top:2px;">
                <a href="https://ultimosismo.igp.gob.pe/" target="_blank" class="btn-igp-mini" style="color:#58a6ff; font-size:0.65rem; text-decoration:none;">
                    🌐 Ver en IGP ↗
                </a>
            </div>
        </div>
    `;
    const lastIcon = L.divIcon({ className: 'custom-div-icon', html: `<div style="background:rgba(248,81,73,0.3); width:40px; height:40px; border-radius:50%; border:2px solid #f85149; animation:pulse-red 2s infinite;"></div>`, iconSize: [40, 40], iconAnchor: [20, 20] });
    const popup = `<div class="popup-dark"><div class="popup-header">🔴 Último Sismo</div><div class="popup-row"><span class="popup-icon">📍</span> ${event.place}</div><div class="popup-row"><span class="popup-icon">⚡</span> Mag: <strong>${event.magnitude.toFixed(1)}</strong></div><div class="popup-row"><span class="popup-icon">🕒</span> ${formatDatePE(event.event_time_utc).split(',')[1]}</div><div class="popup-row"><span class="popup-icon">📉</span> Prof: <strong>${event.depth_km} km</strong></div></div>`;
    L.marker([event.lat, event.lon], { icon: lastIcon }).addTo(map).bindPopup(popup);
});

// 3. Panel
fetch('/api/forecast/topk').then(r => r.json()).then(data => {
    document.getElementById('generated-date').innerText = formatDatePE(data.generated_at);
    document.getElementById('input-date').innerText = new Date(data.input_max_time).toLocaleDateString('es-PE', {timeZone: 'America/Lima'});
    const vStart = new Date(data.topk[0].t_pred_start); const vEnd = new Date(data.topk[0].t_pred_end);
    document.getElementById('validity-range').innerText = `${vStart.toLocaleDateString('es-PE', {day:'numeric',month:'short'})} al ${vEnd.toLocaleDateString('es-PE', {day:'numeric',month:'short'})}`;

    const maxProb = data.topk[0].prob;
    const recBox = document.getElementById('recommendation-box');
    if (maxProb > 0.85) { recBox.style.display = 'flex'; recBox.classList.add('critical'); document.getElementById('recommendation-text').innerHTML = "<strong>NIVEL CRÍTICO:</strong> Probabilidad > 85%. Revisar planes."; }
    else if (maxProb > 0.70) { recBox.style.display = 'flex'; document.getElementById('recommendation-text').innerHTML = "<strong>PRECAUCIÓN:</strong> Actividad anómala detectada."; }

    const list = document.getElementById('topk-list'); list.innerHTML = ''; 
    data.topk.forEach((item, index) => {
        const probPct = (item.prob * 100).toFixed(1); const rank = index + 1; const riskColor = getColor(item.prob);
        const div = document.createElement('div'); div.className = 'pred-item';
        div.innerHTML = `<div class="rank-num" style="color:${riskColor}; text-shadow:0 0 10px ${riskColor}40;">#${rank}</div><div class="location-info"><div style="font-weight:bold; color:#fff; font-size:0.9rem; margin-bottom:2px;">${item.place} <a href="https://www.google.com/maps/search/?api=1&query=${item.lat},${item.lon}" target="_blank" style="text-decoration:none;">📍</a></div><div class="coords" style="color:#8b949e; font-size:0.75rem;">${item.lat.toFixed(2)}, ${item.lon.toFixed(2)}</div><span class="mag-badge">Mag Est: ≥ ${item.mag_pred}</span></div><div class="prob-container"><div class="prob-val" style="color:${riskColor}">${probPct}%</div><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${probPct}%; background-color:${riskColor}"></div></div></div>`;
        
        div.onclick = (e) => { 
            if(e.target.tagName === 'A') return; 
            if(window.innerWidth < 768) switchTab('map');
            map.flyTo([item.lat, item.lon], 10); 
        };
        list.appendChild(div);
    });
});

// 4. Leyenda
const legend = L.control({ position: 'bottomleft' });
legend.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'info legend');
    const grades = [{ label: 'Crítico (>95%)', color: '#ff0000' }, { label: 'Muy Alto (>85%)', color: '#ff4d4d' }, { label: 'Alto (>70%)', color: '#ff9f1c' }, { label: 'Moderado (>55%)', color: '#ffeb3b' }];
    let html = '<h6>Nivel de Riesgo</h6>';
    grades.forEach(i => { html += `<div class="legend-item"><i class="legend-icon" style="background:${i.color}"></i><span>${i.label}</span></div>`; });
    html += '<div style="margin-top:8px; font-size:0.65rem; color:#8b949e">Prob. de Sismo ≥ M4.0</div>';
    div.innerHTML = html; return div;
};
legend.addTo(map);

if (window.innerWidth < 768) { switchTab('list'); }