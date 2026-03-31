// Frontend logic for BitBat Oracle
// Polls the FastAPI backend endpoints to get real-time info

const API_BASE = window.location.origin;

// DOM Elements
const sysStatusDiv = document.getElementById('system-status');
const dirEl = document.getElementById('prediction-direction');
const confEl = document.getElementById('prediction-confidence');
const confCaptionEl = document.getElementById('confidence-caption');
const timeEl = document.getElementById('prediction-time');
const summaryEl = document.getElementById('prediction-summary');
const probabilityStatusEl = document.getElementById('probability-status');
const probabilityUpEl = document.getElementById('probability-up');
const probabilityFlatEl = document.getElementById('probability-flat');
const probabilityDownEl = document.getElementById('probability-down');
const probabilityUpLabelEl = document.getElementById('probability-up-label');
const probabilityFlatLabelEl = document.getElementById('probability-flat-label');
const probabilityDownLabelEl = document.getElementById('probability-down-label');
const ingestStatus = document.getElementById('ingest-status');
const presetSelector = document.getElementById('preset-selector');
const freqSelector = document.getElementById('freq-selector');
const horizonSelector = document.getElementById('horizon-selector');
const trainButton = document.getElementById('train-button');
const nextPredTime = document.getElementById('next-pred-time');
const eventLog = document.getElementById('event-log');
const statusIndicator = document.getElementById('status-indicator');
const logRefreshStatus = document.getElementById('log-refresh-status');

const POLL_INTERVAL = 3000;
let currentFreqMs = 5 * 60 * 1000; // default 5m
let currentHorizonMs = 30 * 60 * 1000; // default 30m

function parseTimeMs(fStr) {
    if(!fStr) return 5 * 60 * 1000;
    const val = parseInt(fStr.slice(0, -1));
    const unit = fStr.slice(-1);
    if(unit === 'm') return val * 60 * 1000;
    if(unit === 'h') return val * 60 * 60 * 1000;
    if(unit === 'd') return val * 24 * 60 * 60 * 1000;
    return 5 * 60 * 1000;
}

function updateCountdown() {
    const now = Date.now();
    const msToNext = currentFreqMs - (now % currentFreqMs);
    const totalSecs = Math.floor(msToNext / 1000);
    const h = Math.floor(totalSecs / 3600);
    const m = Math.floor((totalSecs % 3600) / 60);
    const s = totalSecs % 60;
    
    if (h > 0) {
        nextPredTime.innerText = `in ${h}h ${m}m ${s}s`;
    } else {
        nextPredTime.innerText = `in ${m}m ${s}s`;
    }

    const cbContainer = document.querySelector('.crystal-ball-container');
    if (msToNext <= 3000 || s === 0 && m === 0 && h === 0) {
        cbContainer.classList.add('casting-active');
        nextPredTime.innerText = "Casting...";
    } else {
        cbContainer.classList.remove('casting-active');
    }
}
setInterval(updateCountdown, 1000);

function clampPercent(value) {
    return Math.max(0, Math.min(value, 100));
}

function formatPercent(value) {
    return `${clampPercent(value).toFixed(1)}%`;
}

function directionLabel(direction) {
    const normalized = String(direction || 'flat').toLowerCase();
    if (normalized === 'up') return 'Bull';
    if (normalized === 'down') return 'Bear';
    return 'Flat';
}

function deriveProbabilities(predData) {
    const rawUp = Number(predData.p_up ?? 0);
    const rawDown = Number(predData.p_down ?? 0);
    let rawFlat = predData.p_flat;

    if (rawFlat == null && (rawUp > 0 || rawDown > 0)) {
        rawFlat = Math.max(0, 1 - rawUp - rawDown);
    }

    const up = clampPercent(rawUp * 100);
    const down = clampPercent(rawDown * 100);
    const flat = clampPercent(Number(rawFlat ?? 0) * 100);
    const total = up + down + flat;

    if (total <= 0) {
        return null;
    }

    if (Math.abs(total - 100) < 0.2) {
        return { up, flat, down };
    }

    return {
        up: (up / total) * 100,
        flat: (flat / total) * 100,
        down: (down / total) * 100,
    };
}

function updateProbabilityUI(probabilities) {
    if (!probabilities) {
        probabilityUpEl.style.width = '0%';
        probabilityFlatEl.style.width = '0%';
        probabilityDownEl.style.width = '0%';
        probabilityUpLabelEl.innerText = 'Bull --';
        probabilityFlatLabelEl.innerText = 'Flat --';
        probabilityDownLabelEl.innerText = 'Bear --';
        probabilityStatusEl.innerText = 'Awaiting probability payload';
        return;
    }

    probabilityUpEl.style.width = `${probabilities.up}%`;
    probabilityFlatEl.style.width = `${probabilities.flat}%`;
    probabilityDownEl.style.width = `${probabilities.down}%`;
    probabilityUpLabelEl.innerText = `Bull ${formatPercent(probabilities.up)}`;
    probabilityFlatLabelEl.innerText = `Flat ${formatPercent(probabilities.flat)}`;
    probabilityDownLabelEl.innerText = `Bear ${formatPercent(probabilities.down)}`;
    probabilityStatusEl.innerText =
        `${formatPercent(probabilities.up)} / ${formatPercent(probabilities.flat)} / ${formatPercent(probabilities.down)}`;
}

function renderLogs(logs) {
    eventLog.innerHTML = '';

    if (!logs || logs.length === 0) {
        const li = document.createElement('li');
        li.className = 'log-entry empty';
        li.textContent = 'No system events yet. Start training or monitoring to populate the grimoire.';
        eventLog.appendChild(li);
        return;
    }

    logs.forEach((ev, index) => {
        const li = document.createElement('li');
        li.className = `log-entry${index === 0 ? ' latest' : ''}`;

        let icon = '🔮';
        const msg = String(ev.message || '').toLowerCase();
        const service = String(ev.service || '').toLowerCase();
        if (ev.level === 'ERROR') icon = '❌';
        else if (ev.level === 'WARNING') icon = '⚠️';
        else if (msg.includes('download') || msg.includes('ingest')) icon = '🔭';
        else if (msg.includes('train') || service.includes('training')) icon = '🔥';
        else if (msg.includes('predict')) icon = '✨';

        const stamp = ev.timestamp
            ? new Date(String(ev.timestamp).endsWith('Z') ? ev.timestamp : `${ev.timestamp}Z`)
            : null;
        const timeText = stamp && !Number.isNaN(stamp.getTime())
            ? stamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            : 'unknown time';
        const serviceText = ev.service ? String(ev.service).replaceAll('_', ' ') : 'system';

        li.innerHTML = `
            <div class="log-meta">
                <span class="log-icon">${icon}</span>
                <span class="log-service">${serviceText}</span>
                <span class="log-time">${timeText}</span>
            </div>
            <div class="log-message">${ev.message}</div>
        `;
        eventLog.appendChild(li);
    });
}

async function fetchSystemData() {
    try {
        const healthRes = await fetch(`${API_BASE}/health/detailed`);
        if(healthRes.ok) {
            const healthData = await healthRes.json();
            if(healthData.status === "ok") {
                sysStatusDiv.innerText = "Oracle is Active";
                statusIndicator.classList.add("pulse-active");
            } else {
                sysStatusDiv.innerText = "System Degraded";
                statusIndicator.classList.remove("pulse-active");
            }
        } else {
            sysStatusDiv.innerText = "Slumbering";
            statusIndicator.classList.remove("pulse-active");
        }

        const ingestRes = await fetch(`${API_BASE}/system/ingestion-status`);
        if(ingestRes.ok) {
            const ingestData = await ingestRes.json();
            const badPrices = ingestData.prices && ingestData.prices.includes('missing');
            ingestStatus.innerText = badPrices ? "Stale 🌫️" : "Synchronized ✨";
        }

        const logsRes = await fetch(`${API_BASE}/system/logs?limit=8`);
        if(logsRes.ok) {
            const logsData = await logsRes.json();
            renderLogs(logsData.logs || []);
            logRefreshStatus.innerText =
                `Live updates every few seconds • refreshed ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
        } else {
            logRefreshStatus.innerText = 'Live updates paused — log feed unavailable';
        }

        const settingsRes = await fetch(`${API_BASE}/system/settings`);
        if(settingsRes.ok) {
            const settingsData = await settingsRes.json();

            // Setup options dynamically
            if(freqSelector.options.length <= 4 && settingsData.valid_freqs) {
                freqSelector.innerHTML = "";
                settingsData.valid_freqs.forEach(f => freqSelector.add(new Option(f, f)));
            }
            if(horizonSelector.options.length <= 4 && settingsData.valid_horizons) {
                horizonSelector.innerHTML = "";
                settingsData.valid_horizons.forEach(h => horizonSelector.add(new Option(h, h)));
            }

            if (document.activeElement !== presetSelector) {
                presetSelector.value = settingsData.preset || 'balanced';
            }
            if (document.activeElement !== freqSelector) {
                freqSelector.value = settingsData.freq || '5m';
            }
            if (document.activeElement !== horizonSelector) {
                horizonSelector.value = settingsData.horizon || '30m';
            }
            
            if (settingsData.freq) {
                currentFreqMs = parseTimeMs(settingsData.freq);
            }
            if (settingsData.horizon) {
                currentHorizonMs = parseTimeMs(settingsData.horizon);
            }
        }

    } catch(err) {
        console.error("Failed to fetch system status:", err);
        sysStatusDiv.innerText = "Oracle Disconnected";
        statusIndicator.classList.remove("pulse-active");
    }
}

async function fetchPredictions() {
    try {
        const freq = freqSelector.value || '5m';
        const horizon = horizonSelector.value || '30m';
        const predRes = await fetch(`${API_BASE}/predictions/latest?freq=${freq}&horizon=${horizon}`);
        if(predRes.ok) {
            const predData = await predRes.json();
            updatePredictionUI(predData);
        } else {
            setEmptyPredictionUI();
        }
    } catch(err) {
        console.error("Failed to fetch predictions:", err);
        setEmptyPredictionUI();
    }
}

function setEmptyPredictionUI() {
    dirEl.innerText = "WAIT";
    dirEl.className = "direction flat";
    confEl.innerText = "--";
    confCaptionEl.innerText = 'Model confidence in this direction';
    timeEl.innerText = "Gathering energy... Need training.";
    summaryEl.innerText =
        'The crystal ball will show the top scenario once the API has a fresh prediction.';
    updateProbabilityUI(null);
}

function updatePredictionUI(predData) {
    if(!predData || Object.keys(predData).length === 0) {
        setEmptyPredictionUI();
        return;
    }

    const direction = predData.predicted_direction || predData.direction || "flat";
    const probabilities = deriveProbabilities(predData);
    const confidenceVal = predData.confidence != null
        ? clampPercent(Number(predData.confidence) * 100)
        : probabilities
            ? Math.max(probabilities.up, probabilities.flat, probabilities.down)
            : null;

    dirEl.innerText = direction.toUpperCase();
    dirEl.className = `direction ${direction.toLowerCase()}`;
    confEl.innerText = confidenceVal == null ? 'n/a' : formatPercent(confidenceVal);
    confCaptionEl.innerText = confidenceVal == null
        ? 'Probability data not available yet'
        : `Model confidence in this ${directionLabel(direction).toLowerCase()} call`;
    summaryEl.innerText = confidenceVal == null
        ? 'This prediction does not include a full probability payload yet.'
        : `${directionLabel(direction)} is the top scenario. The split below shows bullish, flat, and bearish odds.`;
    updateProbabilityUI(probabilities);

    let glowColor = "rgba(180, 133, 255, 0.4)";
    if(direction === 'up') glowColor = "var(--up-glow)";
    else if(direction === 'down') glowColor = "var(--down-glow)";
    else glowColor = "var(--flat-glow)";
    
    document.querySelector('.ball-glow').style.background = `radial-gradient(circle, ${glowColor} 0%, transparent 70%)`;

    let tsStr = predData.timestamp_utc;
    if (tsStr && !tsStr.endsWith("Z")) {
        tsStr += "Z";
    }
    const targetDate = new Date(new Date(tsStr).getTime() + currentHorizonMs);
    const timestamp = targetDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    timeEl.innerText = `Forecast for ${timestamp}`;
}

function createSparkle() {
    const star = document.createElement('div');
    star.style.position = 'absolute';
    star.style.width = '2px';
    star.style.height = '2px';
    star.style.background = 'white';
    star.style.borderRadius = '50%';
    star.style.boxShadow = '0 0 5px white';
    star.style.left = Math.random() * window.innerWidth + 'px';
    star.style.top = Math.random() * window.innerHeight + 'px';
    star.style.animation = `blink ${Math.random() * 2 + 1}s linear forwards`;
    document.getElementById('stars').appendChild(star);
    setTimeout(() => star.remove(), 3000);
}

async function updateSetting(payload) {
    try {
        await fetch(`${API_BASE}/system/settings`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        fetchSystemData(); // visually refresh
    } catch (err) {
        console.error("Failed to update settings:", err);
    }
}

presetSelector.addEventListener('change', (e) => updateSetting({ preset: e.target.value }));
freqSelector.addEventListener('change', (e) => updateSetting({ freq: e.target.value }));
horizonSelector.addEventListener('change', (e) => updateSetting({ horizon: e.target.value }));

trainButton.addEventListener('click', async () => {
    trainButton.disabled = true;
    trainButton.innerText = "Summoning... 🔮";
    try {
        await fetch(`${API_BASE}/system/training/start`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ preset: presetSelector.value })
        });
    } catch (err) {
        console.error("Failed to start training:", err);
    } finally {
        trainButton.disabled = false;
        trainButton.innerText = "Summon Oracle (Train)";
    }
});

setInterval(createSparkle, 500);

setInterval(() => {
    fetchSystemData();
    fetchPredictions();
}, POLL_INTERVAL);

fetchSystemData();
fetchPredictions();
