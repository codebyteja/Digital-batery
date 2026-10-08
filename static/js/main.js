/**
 * ==========================================================================
 * main.js — Frontend Logic for EV Battery Digital Twin Dashboard
 * ==========================================================================
 * Handles:
 *   - Tab navigation
 *   - Battery prediction (SoH & RUL) via /predict API
 *   - Historical SoH chart rendering (Chart.js)
 *   - Charging advisor via /advisor API
 *   - Cost estimator via /cost API
 *   - Chatbot widget via /chatbot API
 *   - PDF report upload and auto-fill via /upload_report API
 *   - Loading spinners and animations
 * ==========================================================================
 */

// ── Global State ──
let sohChart = null;           // Chart.js instance for SoH history
let costChart = null;          // Chart.js instance for cost comparison
let gaugeSocChart = null;      // Chart.js doughnut for SoC
let gaugeSohChart = null;      // Chart.js doughnut for SoH
let gaugeRulChart = null;      // Chart.js doughnut for RUL
let gaugeTempChart = null;     // Chart.js doughnut for Temp
let lastPrediction = null;     // Store last prediction for chatbot context
let lastDefectResult = null;   // Store last visual defect scan for chatbot context
let chatbotOpen = false;       // Chatbot window state

// ── DOM Ready ──
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initChatbot();
    initFileUpload();
    initDefectUpload();
    loadSampleReports();
    loadModelMetrics();

    // Auto-select first battery if dropdown exists
    const batterySelect = document.getElementById('batterySelect');
    if (batterySelect && batterySelect.options.length > 1) {
        batterySelect.selectedIndex = 1; // Skip the placeholder
        onBatterySelect();
    }
});


// =========================================================================
// TAB NAVIGATION
// =========================================================================

/**
 * Initialize tab switching. Clicking a tab hides all panels
 * and shows only the selected one.
 */
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and panels
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

            // Activate clicked tab and its panel
            tab.classList.add('active');
            const panelId = tab.getAttribute('data-tab');
            document.getElementById(panelId).classList.add('active');
        });
    });
}

/**
 * Switch to a specific tab programmatically (e.g., from PDF upload).
 */
function switchToTab(tabName) {
    const tab = document.querySelector(`.nav-tab[data-tab="${tabName}"]`);
    if (tab) tab.click();
}


// =========================================================================
// BATTERY PREDICTION
// =========================================================================

/**
 * Called when user selects a battery from the dropdown.
 * Loads the first cycle's values as defaults and fetches history.
 */
function onBatterySelect() {
    const batteryId = document.getElementById('batterySelect').value;
    if (!batteryId) return;

    // Fetch battery history to get sample values
    fetch(`/battery_history/${batteryId}`)
        .then(res => res.json())
        .then(data => {
            if (data.error) return;

            // Set form inputs to roughly mid-life values for a good demo
            const midIdx = Math.floor(data.cycles.length / 2);
            document.getElementById('voltageInput').value =
                data.voltage_values ? data.voltage_values[midIdx].toFixed(2) : '3.5';
            document.getElementById('currentInput').value = '1.0';
            document.getElementById('temperatureInput').value =
                data.temperature_values ? data.temperature_values[midIdx].toFixed(1) : '30';
            document.getElementById('cycleInput').value = data.cycles[midIdx];

            // Also plot the history chart
            plotSoHHistory(data, null);
        })
        .catch(err => console.error('Error loading battery history:', err));
}

/**
 * Run prediction by calling the /predict endpoint.
 * Displays SoC, SoH gauge, RUL, and status badge.
 */
function runPrediction() {
    const voltage = parseFloat(document.getElementById('voltageInput').value);
    const current = parseFloat(document.getElementById('currentInput').value);
    const temperature = parseFloat(document.getElementById('temperatureInput').value);
    const cycleNumber = parseInt(document.getElementById('cycleInput').value);

    // Validate inputs
    if (isNaN(voltage) || isNaN(current) || isNaN(temperature) || isNaN(cycleNumber)) {
        showToast('Please fill in all input fields with valid numbers.', 'warning');
        return;
    }

    // Show loading state on button
    const btn = document.getElementById('predictBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voltage, current, temperature, cycle_number: cycleNumber })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        // Store for chatbot
        lastPrediction = data;

        // Show results panel
        document.getElementById('resultsPanel').classList.add('visible');

        // Update SoC display (estimated from voltage — simple linear mapping)
        const soc = Math.min(100, Math.max(0, ((voltage - 2.5) / (4.2 - 2.5)) * 100));
        
        // Render all 4 gauges
        renderGauge('gaugeSoc', soc, 'socCenter', '%', ['#3b82f6', 'rgba(255, 255, 255, 0.1)'], 100);
        
        let sohColor = '#ef4444';
        if (data.soh > 85) sohColor = '#22c55e';
        else if (data.soh >= 60) sohColor = '#f59e0b';
        renderGauge('gaugeSoh', data.soh, 'sohCenter', '%', [sohColor, 'rgba(255, 255, 255, 0.1)'], 100);

        // Max cycles roughly 2000 for gauge scale
        renderGauge('gaugeRul', data.rul_cycles, 'rulCenter', '', ['#f97316', 'rgba(255, 255, 255, 0.1)'], 2000, cycleNumber);
        const rulYearsElem = document.getElementById('rulYearsLabel');
        if (rulYearsElem) rulYearsElem.innerHTML = `Full cycles left<br><span style="color:var(--text-light)">≈ ${data.rul_years} years of usage</span>`;
        
        let tempColor = temperature > 45 ? '#ef4444' : '#8b5cf6';
        renderGauge('gaugeTemp', temperature, 'tempCenter', '°C', [tempColor, 'rgba(255, 255, 255, 0.1)'], 60);
        document.getElementById('tempWarning').style.display = temperature > 45 ? 'inline' : 'none';

        // Update status badge
        const badge = document.getElementById('statusBadge');
        badge.className = `status-badge ${data.status_class}`;
        const statusIcons = { healthy: '✅', monitor: '⚠️', service: '🔴' };
        badge.innerHTML = `${statusIcons[data.status_class] || '❓'} ${data.status}`;

        // Auto-fill cost estimator with current SoH
        const costSohInput = document.getElementById('costSohInput');
        if (costSohInput) costSohInput.value = data.soh;

        // Load and plot battery history with prediction point
        const batteryId = document.getElementById('batterySelect').value;
        if (batteryId) {
            fetch(`/battery_history/${batteryId}`)
                .then(res => res.json())
                .then(histData => {
                    if (!histData.error) {
                        plotSoHHistory(histData, {
                            cycle: cycleNumber,
                            soh: data.soh
                        });
                    }
                });
        }
    })
    .catch(err => {
        console.error('Prediction error:', err);
        showToast('Prediction failed. Is the server running?', 'error');
    })
    .finally(() => {
        btn.classList.remove('loading');
        btn.disabled = false;
    });
}

/**
 * Render a circular gauge chart using Chart.js doughnut
 */
function renderGauge(canvasId, value, centerId, unit, colors, maxScale, usedValue = null) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    // Determine the variable name for the chart instance
    const chartVarMap = {
        'gaugeSoc': 'gaugeSocChart',
        'gaugeSoh': 'gaugeSohChart',
        'gaugeRul': 'gaugeRulChart',
        'gaugeTemp': 'gaugeTempChart'
    };
    const chartVar = chartVarMap[canvasId];
    
    if (window[chartVar]) {
        window[chartVar].destroy();
    }
    
    let chartData = [value, Math.max(0, maxScale - value)];
    if (usedValue !== null) {
        chartData = [value, usedValue]; // For RUL: remaining vs used
    }

    const roundedVal = (canvasId === 'gaugeRul') ? Math.round(value) : value.toFixed(1);
    document.getElementById(centerId).textContent = `${roundedVal}${unit}`;

    window[chartVar] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: chartData,
                backgroundColor: colors,
                borderWidth: 0,
                cutout: '80%',
                circumference: 360,
                rotation: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                animateScale: true,
                animateRotate: true,
                duration: 1000,
                easing: 'easeOutQuart'
            },
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });
}


// =========================================================================
// SOH HISTORY CHART (Chart.js)
// =========================================================================

/**
 * Plot SoH degradation trend over cycles using Chart.js.
 * Optionally highlights the current prediction point.
 *
 * @param {Object} histData - { cycles: [...], soh_values: [...] }
 * @param {Object|null} predPoint - { cycle: N, soh: M } or null
 */
function plotSoHHistory(histData, predPoint) {
    const ctx = document.getElementById('sohHistoryChart');
    if (!ctx) return;

    // Destroy previous chart instance
    if (sohChart) {
        sohChart.destroy();
    }

    const datasets = [
        {
            label: `${histData.battery_id} — SoH Trend`,
            data: histData.cycles.map((c, i) => ({ x: c, y: histData.soh_values[i] })),
            borderColor: 'rgba(59, 130, 246, 0.8)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 1.5,
            pointHoverRadius: 5,
            borderWidth: 2,
        }
    ];

    // Add prediction point marker if provided
    if (predPoint) {
        datasets.push({
            label: 'Current Prediction',
            data: [{ x: predPoint.cycle, y: predPoint.soh }],
            backgroundColor: '#f59e0b',
            borderColor: '#f59e0b',
            pointRadius: 8,
            pointHoverRadius: 12,
            pointStyle: 'star',
            showLine: false,
        });
    }

    // Add EOL threshold line (80%)
    datasets.push({
        label: 'End-of-Life Threshold (80%)',
        data: [
            { x: histData.cycles[0], y: 80 },
            { x: histData.cycles[histData.cycles.length - 1], y: 80 }
        ],
        borderColor: 'rgba(239, 68, 68, 0.5)',
        borderDash: [8, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
    });

    sohChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'nearest'
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Cycle Number',
                        color: '#94a3b8',
                        font: { size: 12, family: 'Inter' }
                    },
                    ticks: { color: '#64748b', font: { size: 11 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
                y: {
                    title: {
                        display: true,
                        text: 'State of Health (%)',
                        color: '#94a3b8',
                        font: { size: 12, family: 'Inter' }
                    },
                    ticks: { color: '#64748b', font: { size: 11 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    min: 50,
                    max: 105,
                },
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' },
                        usePointStyle: true,
                        padding: 16,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'Inter' },
                }
            }
        }
    });
}


// =========================================================================
// CHARGING ADVISOR
// =========================================================================

/**
 * Fetch charging recommendations from /advisor endpoint.
 */
function getAdvisorRecommendation() {
    const dailyKm = parseFloat(document.getElementById('dailyDistanceInput').value);

    if (isNaN(dailyKm) || dailyKm <= 0) {
        showToast('Please enter a valid daily driving distance.', 'warning');
        return;
    }

    const btn = document.getElementById('advisorBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    fetch('/advisor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ daily_distance_km: dailyKm })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        // Show results
        document.getElementById('advisorResults').classList.add('visible');

        // Update recommended SoC range callout
        document.getElementById('socRangeMin').textContent = `${data.recommended_soc_min}%`;
        document.getElementById('socRangeMax').textContent = `${data.recommended_soc_max}%`;
        document.getElementById('usageLevel').textContent = data.usage_level;

        // Build comparison table
        const tbody = document.getElementById('advisorTableBody');
        tbody.innerHTML = '';

        data.comparison.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight:600">${row.method}</td>
                <td><span class="tag ${row.degradation_percent_per_year > 2 ? 'red' : row.degradation_percent_per_year > 1.5 ? 'yellow' : 'green'}">${row.degradation_percent_per_year}%/yr</span></td>
                <td>${row.charging_time}</td>
                <td>${row.estimated_lifespan_years} years</td>
                <td style="font-size:0.8rem;color:var(--text-secondary)">${row.recommendation}</td>
            `;
            tbody.appendChild(tr);
        });
    })
    .catch(err => {
        console.error('Advisor error:', err);
        showToast('Failed to get advisor recommendation.', 'error');
    })
    .finally(() => {
        btn.classList.remove('loading');
        btn.disabled = false;
    });
}


// =========================================================================
// COST ESTIMATOR
// =========================================================================

/**
 * Calculate cost impact and savings from /cost endpoint.
 */
function calculateCost() {
    const replacementCost = parseFloat(document.getElementById('costReplacementInput').value);
    const currentSoh = parseFloat(document.getElementById('costSohInput').value);

    if (isNaN(replacementCost) || isNaN(currentSoh)) {
        showToast('Please fill in both fields.', 'warning');
        return;
    }

    const btn = document.getElementById('costBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    fetch('/cost', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ replacement_cost: replacementCost, current_soh: currentSoh })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        document.getElementById('costResults').classList.add('visible');

        // Update cost impact display
        document.getElementById('costImpactValue').textContent = `$${data.cost_impact.toLocaleString()}`;
        document.getElementById('savingsValue').textContent = `$${data.savings_3yr.toLocaleString()}`;
        document.getElementById('current3yrSoh').textContent = `${data.current_3yr_soh}%`;
        document.getElementById('optimized3yrSoh').textContent = `${data.optimized_3yr_soh}%`;

        // Update cost comparison bars
        const maxCost = Math.max(data.current_3yr_cost, data.optimized_3yr_cost, 1);
        const currentBarHeight = (data.current_3yr_cost / maxCost) * 100;
        const optimizedBarHeight = (data.optimized_3yr_cost / maxCost) * 100;

        const currentBar = document.getElementById('currentCostBar');
        const optimizedBar = document.getElementById('optimizedCostBar');

        currentBar.style.height = `${Math.max(currentBarHeight, 10)}%`;
        currentBar.textContent = `$${data.current_3yr_cost.toLocaleString()}`;

        optimizedBar.style.height = `${Math.max(optimizedBarHeight, 10)}%`;
        optimizedBar.textContent = `$${data.optimized_3yr_cost.toLocaleString()}`;
    })
    .catch(err => {
        console.error('Cost error:', err);
        showToast('Failed to calculate cost.', 'error');
    })
    .finally(() => {
        btn.classList.remove('loading');
        btn.disabled = false;
    });
}


// =========================================================================
// PDF REPORT UPLOAD
// =========================================================================

/**
 * Initialize the file upload zone with drag-and-drop support.
 */
function initFileUpload() {
    // 1. Existing Upload Report Zone
    const dropZone = document.getElementById('uploadZone');
    if (dropZone) {
        const fileInput = document.getElementById('reportFileInput');
        ['dragenter', 'dragover'].forEach(event => {
            dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length > 0) uploadReport(fileInput.files[0]); });
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) uploadReport(files[0]);
        });
    }

    // 2. NEW: Universal File Upload Zone (Dashboard)
    const uniZone = document.getElementById('universalUploadZone');
    const uniInput = document.getElementById('universalFileInput');
    if (uniZone && uniInput) {
        uniZone.addEventListener('click', () => uniInput.click());
        ['dragenter', 'dragover'].forEach(event => {
            uniZone.addEventListener(event, (e) => { e.preventDefault(); uniZone.style.borderColor = '#3b82f6'; uniZone.style.backgroundColor = 'rgba(59, 130, 246, 0.05)'; });
        });
        ['dragleave', 'drop'].forEach(event => {
            uniZone.addEventListener(event, (e) => { e.preventDefault(); uniZone.style.borderColor = 'var(--border-color)'; uniZone.style.backgroundColor = 'transparent'; });
        });
        uniInput.addEventListener('change', () => { if (uniInput.files.length > 0) handleUniversalUpload(uniInput.files[0]); });
        uniZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) handleUniversalUpload(files[0]);
        });
    }
}

function handleUniversalUpload(file) {
    const statusDiv = document.getElementById('universalUploadStatus');
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '<span class="spinner" style="display:inline-block;width:14px;height:14px;margin-right:6px"></span> Processing file...';
    
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload_universal', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            statusDiv.innerHTML = `<span style="color:#ef4444">❌ Error: ${data.error}</span>`;
            return;
        }

        if (!data.verification.passed) {
            statusDiv.innerHTML = `<span style="color:#ef4444">⚠️ Verification failed — please check file (${data.verification.fields_found} fields found)</span>`;
            return;
        }

        statusDiv.innerHTML = `<span style="color:#22c55e">✅ Data verified successfully</span>`;
        
        // Auto-fill prediction form
        const ext = data.extracted;
        if (ext.voltage) document.getElementById('voltageInput').value = ext.voltage;
        if (ext.current) document.getElementById('currentInput').value = ext.current;
        if (ext.temperature) document.getElementById('temperatureInput').value = ext.temperature;
        if (ext.cycle_number) document.getElementById('cycleInput').value = Math.round(ext.cycle_number);
        
        // Run prediction
        setTimeout(() => runPrediction(), 1000);
    })
    .catch(err => {
        console.error('Universal upload error:', err);
        statusDiv.innerHTML = `<span style="color:#ef4444">❌ Upload failed.</span>`;
    });
}

/**
 * Upload a PDF report to /upload_report and auto-fill the prediction form.
 */
function uploadReport(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('Please upload a PDF file.', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    const uploadStatus = document.getElementById('uploadStatus');
    uploadStatus.innerHTML = '<div class="spinner" style="display:inline-block;margin-right:8px"></div> Processing report...';
    uploadStatus.style.display = 'block';

    fetch('/upload_report', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            uploadStatus.innerHTML = `<span class="text-red">❌ ${data.error}</span>`;
            return;
        }

        const extracted = data.extracted;
        const fields = Object.keys(extracted);

        if (fields.length === 0) {
            uploadStatus.innerHTML = `<span class="text-yellow">⚠️ ${data.message}</span>`;
            return;
        }

        // Show extracted values
        uploadStatus.innerHTML = `<span class="text-green">✅ ${data.message}</span>`;

        const valuesContainer = document.getElementById('extractedValues');
        valuesContainer.innerHTML = '';
        valuesContainer.style.display = 'grid';

        for (const [key, value] of Object.entries(extracted)) {
            const item = document.createElement('div');
            item.className = 'extracted-item';
            item.innerHTML = `
                <div>
                    <div class="label">${key.replace('_', ' ')}</div>
                    <div class="value">${value}</div>
                </div>
            `;
            valuesContainer.appendChild(item);
        }

        // Auto-fill the prediction form
        if (extracted.voltage) document.getElementById('voltageInput').value = extracted.voltage;
        if (extracted.current) document.getElementById('currentInput').value = extracted.current;
        if (extracted.temperature) document.getElementById('temperatureInput').value = extracted.temperature;
        if (extracted.cycle_number) document.getElementById('cycleInput').value = Math.round(extracted.cycle_number);

        // Auto-navigate to dashboard and run prediction after a short delay
        setTimeout(() => {
            switchToTab('dashboard');
            runPrediction();
        }, 1500);
    })
    .catch(err => {
        console.error('Upload error:', err);
        uploadStatus.innerHTML = '<span class="text-red">❌ Upload failed. Check your connection.</span>';
    });
}


// =========================================================================
// SAMPLE REPORTS
// =========================================================================

/**
 * Load list of available sample PDFs from the server.
 */
function loadSampleReports() {
    fetch('/sample_reports_list')
        .then(res => res.json())
        .then(data => {
            const container = document.getElementById('sampleReportsGrid');
            if (!container) return;

            if (data.reports.length === 0) {
                container.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem">No sample reports found. Run: python generate_sample_reports.py</p>';
                return;
            }

            container.innerHTML = '';
            const icons = ['📄', '📋', '📊'];
            const labels = ['Healthy Battery', 'Moderate Wear', 'Degraded Battery'];

            data.reports.forEach((filename, idx) => {
                const card = document.createElement('a');
                card.className = 'sample-report-card';
                card.href = `/sample_reports/${filename}`;
                card.download = filename;
                card.innerHTML = `
                    <div class="report-icon">${icons[idx] || '📄'}</div>
                    <div class="report-info">
                        <h5>${labels[idx] || filename}</h5>
                        <p>${filename}</p>
                    </div>
                `;
                container.appendChild(card);
            });
        })
        .catch(err => console.error('Error loading sample reports:', err));
}


// =========================================================================
// MODEL METRICS (About Page)
// =========================================================================

/**
 * Fetch and display model accuracy metrics.
 */
function loadModelMetrics() {
    fetch('/metrics')
        .then(res => res.json())
        .then(data => {
            // SoH model metrics
            const sohR2 = document.getElementById('sohR2');
            const sohRMSE = document.getElementById('sohRMSE');
            if (sohR2) sohR2.textContent = data.soh_model?.r2 ?? 'N/A';
            if (sohRMSE) sohRMSE.textContent = data.soh_model?.rmse ?? 'N/A';

            // RUL model metrics
            const rulR2 = document.getElementById('rulR2');
            const rulRMSE = document.getElementById('rulRMSE');
            if (rulR2) rulR2.textContent = data.rul_model?.r2 ?? 'N/A';
            if (rulRMSE) rulRMSE.textContent = data.rul_model?.rmse ?? 'N/A';
        })
        .catch(err => console.error('Error loading metrics:', err));
}


// =========================================================================
// CHATBOT
// =========================================================================

/**
 * Initialize chatbot toggle and input handling.
 */
function initChatbot() {
    const toggle = document.getElementById('chatbotToggle');
    const window_ = document.getElementById('chatbotWindow');
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');

    if (!toggle || !window_) return;

    // Toggle chatbot window
    toggle.addEventListener('click', () => {
        chatbotOpen = !chatbotOpen;
        window_.classList.toggle('open', chatbotOpen);
        toggle.classList.toggle('active', chatbotOpen);
        toggle.textContent = chatbotOpen ? '✕' : '💬';

        if (chatbotOpen) {
            input.focus();
            // Send welcome message on first open
            const messages = document.getElementById('chatMessages');
            if (messages.children.length === 0) {
                addChatMessage('bot', '👋 Hi! I\'m your EV Battery Assistant. Ask me anything about battery health, charging tips, or how to extend your battery\'s life!');
            }
            renderChatSuggestions();
        }
    });

    // Send message on Enter key
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });

    sendBtn.addEventListener('click', sendChatMessage);
}

/**
 * Send user message to the chatbot and display the reply.
 */
function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    // Display user message
    addChatMessage('user', message);
    input.value = '';

    // Build context
    const context = {};
    if (lastPrediction) {
        context.soh = lastPrediction.soh;
        context.rul = lastPrediction.rul_cycles;
    }
    if (lastDefectResult) {
        context.defect_result = {
            n_damaged: lastDefectResult.n_damaged || 0,
            n_good: lastDefectResult.n_good || 0,
            severity: lastDefectResult.severity || 'Unknown',
            result: lastDefectResult.result || ''
        };
    }
    
    // Add current form values if they exist
    const v = document.getElementById('voltage')?.value;
    const c = document.getElementById('current')?.value;
    const t = document.getElementById('temperature')?.value;
    if (v) context.voltage = v;
    if (c) context.current = c;
    if (t) context.temperature = t;

    // Call chatbot API
    fetch('/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, context })
    })
    .then(res => res.json())
    .then(data => {
        addChatMessage('bot', data.reply);
    })
    .catch(err => {
        addChatMessage('bot', '😔 Sorry, I couldn\'t connect to the server. Please try again.');
    });
}

/**
 * Add a message bubble to the chat window.
 */
function addChatMessage(sender, text) {
    const messages = document.getElementById('chatMessages');
    const msg = document.createElement('div');
    msg.className = `chat-message ${sender}`;
    msg.textContent = text;
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
}

/**
 * Render quick reply suggestion chips based on available context.
 */
function renderChatSuggestions(mode = 'default') {
    const container = document.getElementById('chatSuggestions');
    if (!container) return;
    container.innerHTML = ''; // clear old suggestions

    let suggestions = [];

    if (mode === 'defect') {
        suggestions = ["How many damaged cells?", "Estimate repair cost", "What is visual defect scanning?"];
    } else if (lastPrediction) {
        suggestions = ["Show my SoH", "Check battery life (RUL)", "How many services should we do?"];
    } else {
        suggestions = ["What is SoH?", "Cost estimator", "How to scan for damage?"];
    }

    suggestions.forEach(text => {
        const chip = document.createElement('div');
        chip.style.cssText = 'padding:6px 12px; background:var(--primary-600); color:white; border-radius:15px; font-size:12px; cursor:pointer; font-weight:500; box-shadow:0 2px 4px rgba(0,0,0,0.15); flex-shrink:0; white-space:nowrap; border:1px solid rgba(255,255,255,0.1);';
        chip.textContent = text;
        chip.onclick = () => {
            const input = document.getElementById('chatInput');
            input.value = text;
            sendChatMessage();
            renderChatSuggestions(); // reset to default
        };
        container.appendChild(chip);
    });
}


// =========================================================================
// TOAST NOTIFICATIONS
// =========================================================================

/**
 * Show a brief toast notification at the top of the screen.
 */
function showToast(message, type = 'info') {
    // Remove existing toast if any
    const existing = document.querySelector('.toast-notification');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast-notification';

    const colors = {
        info: 'var(--primary-500)',
        success: 'var(--accent-500)',
        warning: '#f59e0b',
        error: '#ef4444'
    };

    const icons = {
        info: 'ℹ️',
        success: '✅',
        warning: '⚠️',
        error: '❌'
    };

    toast.style.cssText = `
        position: fixed;
        top: 80px;
        left: 50%;
        transform: translateX(-50%) translateY(-20px);
        background: var(--bg-secondary);
        border: 1px solid ${colors[type]};
        border-radius: var(--radius-sm);
        padding: 12px 24px;
        font-size: 0.85rem;
        color: var(--text-primary);
        z-index: 3000;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        animation: toastIn 0.3s cubic-bezier(0.16,1,0.3,1) forwards;
    `;

    toast.innerHTML = `${icons[type]} ${message}`;
    document.body.appendChild(toast);

    // Add toast animation styles if not already present
    if (!document.getElementById('toastStyles')) {
        const style = document.createElement('style');
        style.id = 'toastStyles';
        style.textContent = `
            @keyframes toastIn {
                from { opacity: 0; transform: translateX(-50%) translateY(-20px); }
                to   { opacity: 1; transform: translateX(-50%) translateY(0); }
            }
            @keyframes toastOut {
                from { opacity: 1; transform: translateX(-50%) translateY(0); }
                to   { opacity: 0; transform: translateX(-50%) translateY(-20px); }
            }
        `;
        document.head.appendChild(style);
    }

    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}


// =========================================================================
// DEFECT DETECTION MODULE
// =========================================================================

function initDefectUpload() {
    const uploadZone = document.getElementById('defectUploadZone');
    const fileInput = document.getElementById('defectFileInput');
    
    if (!uploadZone || !fileInput) return;

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--accent-500)';
        uploadZone.style.background = 'rgba(16, 185, 129, 0.05)';
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--primary)';
        uploadZone.style.background = 'rgba(99,102,241,0.04)';
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--primary)';
        uploadZone.style.background = 'transparent';
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleDefectUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleDefectUpload(e.target.files[0]);
        }
    });
}

function handleDefectUpload(file) {
    const loadingDiv = document.getElementById('defectLoading');
    const resultBox  = document.getElementById('defectResultBox');

    if (!file.name.match(/\.(jpg|jpeg|png|pdf|zip)$/i)) {
        showToast('Please upload a JPG, PNG, PDF, or ZIP file.', 'warning');
        return;
    }

    // Reset UI
    resultBox.style.display = 'none';
    loadingDiv.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/detect_defect', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
        loadingDiv.style.display = 'none';
        if (data.error) { showToast(data.error, 'error'); return; }

        resultBox.style.display = 'block';

        const badge     = document.getElementById('defectResultBadge');
        const actionMsg = document.getElementById('defectActionMsg');
        const conf      = document.getElementById('defectConfidence');
        const imgType   = document.getElementById('defectImageType');
        const bulkInfo  = document.getElementById('defectBulkInfo');
        const breakdown = document.getElementById('defectBreakdown');
        const note      = document.getElementById('defectDemoNote');
        const dlZone    = document.getElementById('defectDownloadZone');
        const dlImg     = document.getElementById('defectCompositeImg');
        const dlBtn     = document.getElementById('defectDownloadBtn');

        const isDamaged = data.result === 'Battery Damaged';

        // ── Main result colouring ──
        if (!isDamaged) {
            resultBox.style.background = 'rgba(16,185,129,0.1)';
            resultBox.style.border     = '1px solid #10b981';
            badge.style.color          = '#10b981';
            badge.innerHTML            = '✅ Battery Good';
        } else {
            resultBox.style.background = 'rgba(239,68,68,0.1)';
            resultBox.style.border     = '1px solid #ef4444';
            badge.style.color          = '#ef4444';
            badge.innerHTML            = '⚠️ Battery Damaged';
        }

        // ── Action / service alert ──
        if (data.action) {
            actionMsg.style.display    = 'block';
            actionMsg.textContent      = data.action;
            actionMsg.style.background = isDamaged ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)';
            actionMsg.style.color      = isDamaged ? '#fca5a5' : '#6ee7b7';
            actionMsg.style.border     = isDamaged ? '1px solid #ef4444' : '1px solid #10b981';
        } else {
            actionMsg.style.display = 'none';
        }

        // ── Stats chips ──
        // For damaged: label confidence as 'Damage Score' so user understands it's not a health %
        const confLabel = isDamaged ? 'Damage Score' : 'Confidence';
        conf.textContent    = `${confLabel}: ${data.confidence}%`;
        imgType.textContent = `Mode: ${data.image_mode}`;

        // Severity chip (only for damaged)
        let severityChip = document.getElementById('defectSeverityChip');
        if (!severityChip) {
            severityChip = document.createElement('span');
            severityChip.id = 'defectSeverityChip';
            severityChip.style.cssText = 'font-size:0.85rem; padding:0.3rem 0.9rem; border-radius:20px; font-weight:700;';
            conf.parentNode.insertBefore(severityChip, conf.nextSibling);
        }
        if (isDamaged && data.severity) {
            const sevColor = data.severity.includes('Severe') ? '#ef4444'
                           : data.severity.includes('Moderate') ? '#f97316' : '#eab308';
            severityChip.textContent    = data.severity;
            severityChip.style.display  = 'inline-block';
            severityChip.style.background = `rgba(${sevColor.includes('4444') ? '239,68,68' : sevColor.includes('7316') ? '249,115,22' : '234,179,8'},0.15)`;
            severityChip.style.color    = sevColor;
            severityChip.style.border   = `1px solid ${sevColor}`;
        } else {
            severityChip.style.display  = 'none';
        }

        if (data.bulk_info) {
            bulkInfo.textContent    = data.bulk_info;
            bulkInfo.style.display  = 'inline-block';
        } else {
            bulkInfo.style.display  = 'none';
        }

        // ── Per-image breakdown ──
        if (data.breakdown && data.breakdown.length > 1) {
            let html = `<div style="font-weight:700; margin-bottom:0.5rem; color:var(--text-light);">Per-Image Results (${data.breakdown.length} analysed):</div>`;
            html += '<table style="width:100%; border-collapse:collapse;">';
            html += '<tr style="color:var(--text-muted); border-bottom:1px solid rgba(255,255,255,0.1);">'
                  + '<th style="padding:3px 6px; text-align:left;">#</th>'
                  + '<th style="padding:3px 6px; text-align:left;">File</th>'
                  + '<th style="padding:3px 6px;">Status</th>'
                  + '<th style="padding:3px 6px;">Score</th>'
                  + '<th style="padding:3px 6px;">Severity</th>'
                  + '<th style="padding:3px 6px;">Mode</th></tr>';
            data.breakdown.forEach((row, i) => {
                const good = row.label === 'Battery Good';
                const col  = good ? '#6ee7b7' : '#fca5a5';
                const icon = good ? '✓' : '✗';
                const sevText = row.severity || '—';
                const sevCol  = row.severity === 'Severe'   ? '#ef4444'
                              : row.severity === 'Moderate' ? '#f97316'
                              : row.severity === 'Mild'     ? '#eab308' : 'var(--text-muted)';
                // Score label: for good=Confidence, for damaged=Damage Score
                html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:3px 6px; color:var(--text-muted); text-align:center;">${i+1}</td>
                    <td style="padding:3px 6px; color:var(--text-secondary); word-break:break-all;">${row.name}</td>
                    <td style="padding:3px 6px; color:${col}; font-weight:600; text-align:center;">${icon} ${row.label}</td>
                    <td style="padding:3px 6px; color:${good ? '#6ee7b7' : '#fca5a5'}; text-align:center; font-weight:600;">${row.confidence}%</td>
                    <td style="padding:3px 6px; color:${sevCol}; text-align:center; font-weight:600;">${sevText}</td>
                    <td style="padding:3px 6px; color:var(--text-muted); text-align:center;">${row.mode}</td>
                </tr>`;
            });
            html += '</table>';
            breakdown.innerHTML = html;
            breakdown.style.display = 'block';
        } else {
            breakdown.style.display = 'none';
        }

        // ── Demo note ──
        note.style.display  = data.is_demo ? 'block' : 'none';
        note.textContent    = data.is_demo ? 'ℹ️ Demo Mode — no trained model found. Results are simulated for demonstration.' : '';

        // ── Composite image download ──
        if (data.composite_b64) {
            dlImg.src        = data.composite_b64;
            dlBtn.href       = data.composite_b64;
            dlZone.style.display = 'block';
        } else {
            dlZone.style.display = 'none';
        }

        // Update Chatbot Context
        lastDefectResult = data;
        renderChatSuggestions('defect');
        
        // Open Chatbot automatically to show the new contextual suggestions
        if (!chatbotOpen) {
            document.getElementById('chatbotToggle').click();
        }

    })
    .catch(err => {
        loadingDiv.style.display = 'none';
        console.error('Defect upload error:', err);
        showToast('Failed to analyse file — check server logs.', 'error');
    });
}


