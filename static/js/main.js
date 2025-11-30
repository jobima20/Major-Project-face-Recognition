// FaceScan Pro - Main JavaScript

let stream = null;
let captureCount = 0;
const MAX_IMAGES = 20;
let currentName = '';

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;

        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update panes
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        document.getElementById(tabName).classList.add('active');

        // Handle webcam when switching away from tabs
        if (tabName === 'train') {
            // Stop live feed when going to train tab
            stopLiveFeed();
            stopLogsRefresh();
        }
    });
});

// Start Live Scan Button
document.getElementById('start-live').addEventListener('click', async () => {
    // Tell backend to start camera
    await fetch('/start_camera', { method: 'POST' });

    startLiveFeed();
    startLogsRefresh();
    document.getElementById('start-live').disabled = true;
    document.getElementById('stop-live').disabled = false;
});

// Stop Live Scan Button
document.getElementById('stop-live').addEventListener('click', async () => {
    // Tell backend to stop camera
    await fetch('/stop_camera', { method: 'POST' });

    stopLiveFeed();
    stopLogsRefresh();
    document.getElementById('start-live').disabled = false;
    document.getElementById('stop-live').disabled = true;
});

// Stop Train Webcam
function stopTrainWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;

        const video = document.getElementById('train-video');
        video.srcObject = null;
    }
}

// Start Live Feed
function startLiveFeed() {
    const liveFeed = document.getElementById('live-feed');
    liveFeed.src = '/video_feed?t=' + new Date().getTime(); // Add timestamp to force refresh
    liveFeed.style.display = 'block';
}

// Stop Live Feed
function stopLiveFeed() {
    const liveFeed = document.getElementById('live-feed');
    liveFeed.src = '';
    liveFeed.style.display = 'none';

    // Reset buttons
    document.getElementById('start-live').disabled = false;
    document.getElementById('stop-live').disabled = true;
}

// Train Tab - Start Capture
document.getElementById('start-capture').addEventListener('click', async () => {
    const nameInput = document.getElementById('person-name');
    currentName = nameInput.value.trim();

    if (!currentName) {
        showStatus('Please enter a person name', 'error');
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });

        const video = document.getElementById('train-video');
        video.srcObject = stream;
        video.play();

        document.getElementById('start-capture').disabled = true;
        document.getElementById('capture-btn').disabled = false;
        nameInput.disabled = true;

        captureCount = 0;
        updateProgress();
        showStatus('Webcam ready! Click "Capture Image" 20 times', 'success');

    } catch (error) {
        showStatus('Could not access webcam: ' + error.message, 'error');
    }
});

// Capture Image
document.getElementById('capture-btn').addEventListener('click', async () => {
    const video = document.getElementById('train-video');
    const canvas = document.getElementById('train-canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('/capture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: currentName, image: imageData })
        });

        const result = await response.json();

        if (result.success) {
            captureCount = result.count;
            updateProgress();
            showStatus(`Captured ${captureCount}/${MAX_IMAGES} (Blur: ${result.blur_score.toFixed(1)})`, 'success');

            if (captureCount >= MAX_IMAGES) {
                await trainModel();
            }
        } else {
            showStatus('Error: ' + result.error, 'error');
        }

    } catch (error) {
        showStatus('Network error: ' + error.message, 'error');
    }
});

// Train Model
async function trainModel() {
    document.getElementById('capture-btn').disabled = true;
    showStatus('Training model... Please wait', 'success');

    try {
        const response = await fetch('/train_model', { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            showStatus(result.message + ' 🎉', 'success');

            // Stop webcam
            stopTrainWebcam();

            // Reset form
            setTimeout(() => {
                document.getElementById('person-name').value = '';
                document.getElementById('person-name').disabled = false;
                document.getElementById('start-capture').disabled = false;
                captureCount = 0;
                updateProgress();
            }, 2000);

        } else {
            showStatus('Training failed: ' + result.error, 'error');
        }

    } catch (error) {
        showStatus('Training error: ' + error.message, 'error');
    }
}

// Update Progress
function updateProgress() {
    const percentage = (captureCount / MAX_IMAGES) * 100;
    document.getElementById('progress-fill').style.width = percentage + '%';
    document.getElementById('progress-text').textContent = `${captureCount} / ${MAX_IMAGES} images`;
}

// Show Status
function showStatus(message, type) {
    const statusEl = document.getElementById('train-status');
    statusEl.textContent = message;
    statusEl.className = `status-message ${type}`;

    if (type === 'error') {
        setTimeout(() => {
            statusEl.className = 'status-message';
        }, 5000);
    }
}

// Logs Refresh
let logsInterval = null;

function startLogsRefresh() {
    loadLogs();
    logsInterval = setInterval(loadLogs, 3000);
}

function stopLogsRefresh() {
    if (logsInterval) {
        clearInterval(logsInterval);
        logsInterval = null;
    }
}

async function loadLogs() {
    try {
        const response = await fetch('/logs');
        const data = await response.json();

        const logsContainer = document.getElementById('logs-list');

        if (data.logs.length === 0) {
            logsContainer.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No detections yet...</p>';
            return;
        }

        logsContainer.innerHTML = data.logs.reverse().map(log => `
            <div class="log-item">
                <div>
                    <span class="log-name">${log.name}</span>
                    <span class="log-confidence">${log.confidence}%</span>
                </div>
                <span class="log-time">${log.timestamp}</span>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load logs:', error);
    }
}

// Clear Logs
document.getElementById('clear-logs').addEventListener('click', async () => {
    try {
        await fetch('/clear_logs', { method: 'POST' });
        document.getElementById('logs-list').innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No detections yet...</p>';
    } catch (error) {
        console.error('Failed to clear logs:', error);
    }
});

// Initialize
updateProgress();
