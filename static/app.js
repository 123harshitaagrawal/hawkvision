let currentTaskId = null;
let currentTelemetry = null;
let pollInterval = null;
let currentVideoSource = 'preset';
let selectedFile = null;
let lastCompletedTaskId = null;
let allHistoryRecords = [];
let currentLoadedHistoryTaskId = null;

// ===== ACUBE AI THEME MANAGEMENT (Light & Dark) =====
function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try {
        localStorage.setItem('hawkvision_theme', theme);
    } catch (e) {}

    const toggleBtn = document.getElementById('themeToggleBtn');
    const toggleText = document.getElementById('themeToggleText');
    if (toggleBtn && toggleText) {
        if (theme === 'light') {
            toggleBtn.classList.add('is-light');
            toggleText.innerText = 'Light';
            toggleBtn.setAttribute('title', 'Switch to Dark Mode');
        } else {
            toggleBtn.classList.remove('is-light');
            toggleText.innerText = 'Dark';
            toggleBtn.setAttribute('title', 'Switch to Light Mode');
        }
    }

    // Refresh 2D pitch map with corresponding theme colors
    if (currentTelemetry) {
        drawPitchTrajectory(currentTelemetry);
    } else {
        initPitchCanvas();
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
}

function initTheme() {
    let saved = 'dark';
    try {
        saved = localStorage.getItem('hawkvision_theme') || 'dark';
    } catch (e) {}
    applyTheme(saved);
}

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    loadPresets();
    initPitchCanvas();
    loadHistory();
    initVideoPlaybackListeners();

    // If loaded from /history with ?load=<task_id>, immediately restore that delivery
    const urlParams = new URLSearchParams(window.location.search);
    const loadTaskId = urlParams.get('load');
    if (loadTaskId) {
        setTimeout(() => {
            loadHistoryItem(loadTaskId);
        }, 400);
    }
});

function switchVideoTab(tab) {
    currentVideoSource = tab;
    const btnPreset = document.getElementById('tabPreset');
    const btnUpload = document.getElementById('tabUpload');
    const secPreset = document.getElementById('presetSection');
    const secUpload = document.getElementById('uploadSection');

    if (tab === 'preset') {
        btnPreset.classList.add('active');
        btnUpload.classList.remove('active');
        secPreset.classList.remove('hidden');
        secUpload.classList.add('hidden');
    } else {
        btnUpload.classList.add('active');
        btnPreset.classList.remove('active');
        secUpload.classList.remove('hidden');
        secPreset.classList.add('hidden');
    }
}

function handleFileSelected(event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        const info = document.getElementById('fileSelectedInfo');
        info.innerText = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
        info.classList.remove('hidden');
    }
}

// Drag & drop
const dropZone = document.getElementById('dropZone');
if (dropZone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#10b981';
            dropZone.style.background = 'rgba(16, 185, 129, 0.1)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '';
            dropZone.style.background = '';
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            document.getElementById('fileInput').files = files;
            handleFileSelected({ target: { files } });
        }
    });
}

async function loadPresets() {
    try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        if (data.sample_videos && data.sample_videos.length > 0) {
            const select = document.getElementById('presetSelect');
            select.innerHTML = '';
            data.sample_videos.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.id;
                opt.textContent = `${v.name} (${v.size_mb} MB)`;
                select.appendChild(opt);
            });
        }
    } catch (e) {
        console.warn('Could not load preset videos:', e);
    }
}

async function startAnalysis() {
    const startBtn = document.getElementById('startBtn');
    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="spinner"></span> Processing Video...';

    const progressBanner = document.getElementById('progressBanner');
    progressBanner.classList.remove('hidden');
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressPercent').innerText = '0%';
    document.getElementById('progressStep').innerText = 'Detecting & Predicting Ball Trajectory...';
    document.getElementById('progFrame').innerText = '0 / 0';
    document.getElementById('progStatus').innerText = 'Searching...';
    document.getElementById('progAngle').innerText = '0.0°';

    const bannerSpinner = progressBanner.querySelector('.spinner');
    if (bannerSpinner) bannerSpinner.style.display = 'block';

    const formData = new FormData();
    formData.append('video_source', currentVideoSource);
    formData.append('model', document.getElementById('modelSelect').value);
    formData.append('conf', document.getElementById('confSlider').value);
    formData.append('history', document.getElementById('historySlider').value);
    formData.append('future', document.getElementById('futureSlider').value);
    formData.append('bezier', document.getElementById('bezierToggle').value || 'true');
    formData.append('auto_trim', document.getElementById('autoTrimToggle').value || 'true');

    if (currentVideoSource === 'upload' && selectedFile) {
        formData.append('file', selectedFile);
    } else {
        formData.append('preset_video', document.getElementById('presetSelect').value);
    }

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.error) {
            alert('Error: ' + data.error);
            resetStartBtn();
            return;
        }

        currentTaskId = data.task_id;
        pollTaskProgress(currentTaskId);

    } catch (err) {
        alert('Analysis request failed: ' + err.message);
        resetStartBtn();
    }
}

function pollTaskProgress(taskId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/task/${taskId}`);
            if (!res.ok) {
                console.warn('Task poll received status:', res.status);
                return;
            }
            const task = await res.json();

            if (task.status === 'processing') {
                const pct = task.progress || 0;
                document.getElementById('progressBar').style.width = pct + '%';
                document.getElementById('progressPercent').innerText = pct + '%';
                document.getElementById('progFrame').innerText = `${task.current_frame || 0} / ${task.total_frames || 1}`;
                
                let statusLabel = task.ball_detected ? 'BALL TRACKED' : 'Searching...';
                if (task.stage === 'segmenting') {
                    statusLabel = 'SEGMENTING DELIVERIES';
                } else if (task.stage === 'tracking') {
                    statusLabel = task.ball_detected ? 'TRACKING BALL' : 'INSPECTING';
                }
                document.getElementById('progStatus').innerText = statusLabel;
                document.getElementById('progAngle').innerText = `${task.angle || 0.0}°`;
                if (task.step_name) {
                    document.getElementById('progressStep').innerText = task.step_name;
                }

            } else if (task.status === 'completed') {
                clearInterval(pollInterval);
                document.getElementById('progressBar').style.width = '100%';
                document.getElementById('progressPercent').innerText = '100%';
                document.getElementById('progressStep').innerText = '✅ Analysis Complete!';

                const totalF = (task.telemetry && task.telemetry.total_frames) || task.total_frames || 1;
                document.getElementById('progFrame').innerText = `${totalF} / ${totalF}`;
                document.getElementById('progStatus').innerText = (task.telemetry && task.telemetry.detected_frames > 0) ? 'BALL TRACKED' : 'Complete';
                document.getElementById('progAngle').innerText = `${task.angle || 0.0}°`;

                const bannerSpinner = document.querySelector('#progressBanner .spinner');
                if (bannerSpinner) bannerSpinner.style.display = 'none';

                setTimeout(() => {
                    document.getElementById('progressBanner').classList.add('hidden');
                    if (bannerSpinner) bannerSpinner.style.display = 'block';
                }, 3500);

                displayResults(task);
                resetStartBtn();

            } else if (task.status === 'failed') {
                clearInterval(pollInterval);
                alert('Inference failed: ' + task.error);
                resetStartBtn();
            }
        } catch (e) {
            console.error('Error polling status:', e);
        }
    }, 500);
}

function resetStartBtn() {
    const startBtn = document.getElementById('startBtn');
    startBtn.disabled = false;
    startBtn.innerHTML = '<span class="btn-icon">⚡</span> Run Trajectory Prediction';
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ==========================================
// VIDEO PLAYBACK & SHOT REPLAY CONTROLLER
// ==========================================
window.currentPlaybackSpeed = 1.0;
window.isShotLooping = false;
window.activeShotBoundary = null; // { shot: 1, start: 0.0, end: 5.03 }

function showVideoHudToast(message, icon = '⚡') {
    const toast = document.getElementById('videoHudToast');
    if (!toast) return;
    toast.innerHTML = `<span style="font-size:1.05rem;">${icon}</span> <span>${escapeHtml(message)}</span>`;
    toast.classList.remove('hidden');
    if (window._videoToastTimer) clearTimeout(window._videoToastTimer);
    window._videoToastTimer = setTimeout(() => {
        toast.classList.add('hidden');
    }, 2400);
}

function setVideoSpeed(speed) {
    const rate = parseFloat(speed) || 1.0;
    window.currentPlaybackSpeed = rate;

    const videoElem = document.getElementById('resultVideo');
    if (videoElem) {
        videoElem.playbackRate = rate;
    }

    // Update active states on all speed buttons across both toolbars
    document.querySelectorAll('.speed-btn, .pitch-speed-pill').forEach(btn => {
        const btnSpeed = parseFloat(btn.getAttribute('data-speed'));
        if (Math.abs(btnSpeed - rate) < 0.01) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    const speedDesc = (rate === 0.25) ? '0.25x Super Slow-Mo' : ((rate === 0.5) ? '0.5x Slow Motion' : '1.0x Normal Speed');
    showVideoHudToast(`Playback Speed: ${speedDesc}`, '⚡');
}

function getShotVideoTiming(shotNum) {
    const shotsData = window.currentShotsData || [];
    const videoElem = document.getElementById('resultVideo');
    const videoDur = videoElem ? (videoElem.duration || 0) : 0;

    if (!shotsData || shotsData.length === 0) {
        return { shot: 1, start: 0.0, end: videoDur, dur: videoDur };
    }

    const sNum = parseInt(shotNum) || 1;
    let accumulated = 0.0;

    for (let i = 0; i < shotsData.length; i++) {
        const s = shotsData[i];
        const dur = s.duration_sec || 3.0;

        if (s.shot === sNum) {
            const vStart = (typeof s.video_start_sec === 'number') ? s.video_start_sec : Math.max(0, accumulated);
            const vEnd = (typeof s.video_end_sec === 'number') ? s.video_end_sec : (accumulated + dur);
            return { shot: sNum, start: vStart, end: vEnd, dur: dur };
        }
        accumulated += dur;
    }

    const first = shotsData[0];
    const firstDur = first.duration_sec || 3.0;
    return { shot: first.shot || 1, start: first.video_start_sec || 0.0, end: first.video_end_sec || firstDur, dur: firstDur };
}

// Reliable Video Loader & Player: guards against Autoplay Policy rejections and async gesture loss
function loadAndPlayVideo(videoElem, targetUrl, startTime = 0.0, autoPlay = true, onLoaded = null) {
    if (!videoElem || !targetUrl) return;

    initVideoPlaybackListeners();

    // Check if source actually needs to change (strip query params for comparison)
    const currentBase = (videoElem.src || '').split('?')[0].replace(window.location.origin, '');
    const targetBase = targetUrl.split('?')[0].replace(window.location.origin, '');
    const needsReload = !currentBase || (currentBase !== targetBase && !currentBase.endsWith(targetBase) && !targetBase.endsWith(currentBase));

    const applyPlayback = () => {
        try {
            if (typeof startTime === 'number' && !isNaN(startTime) && startTime >= 0) {
                videoElem.currentTime = startTime;
            }
        } catch (e) {
            console.warn('Could not set currentTime yet:', e);
        }
        videoElem.playbackRate = window.currentPlaybackSpeed || 1.0;
        if (autoPlay) {
            const playPromise = videoElem.play();
            if (playPromise !== undefined) {
                playPromise.catch(err => {
                    console.warn('Playback deferred or blocked, attempting muted playback:', err);
                    videoElem.muted = true;
                    videoElem.play().catch(e => console.warn('Muted playback also blocked:', e));
                });
            }
        }
        if (typeof onLoaded === 'function') {
            try { onLoaded(); } catch (e) { console.error(e); }
        }
    };

    if (needsReload) {
        videoElem.oncanplay = null;
        videoElem.pause();
        while (videoElem.firstChild) videoElem.removeChild(videoElem.firstChild);
        const source = document.createElement('source');
        source.src = targetUrl;
        source.type = 'video/mp4';
        videoElem.appendChild(source);
        videoElem.src = targetUrl;
        videoElem.load();

        if (videoElem.readyState >= 1) {
            applyPlayback();
        } else {
            const onReady = () => {
                videoElem.removeEventListener('loadedmetadata', onReady);
                videoElem.removeEventListener('canplay', onReady);
                applyPlayback();
            };
            videoElem.addEventListener('loadedmetadata', onReady, { once: true });
            videoElem.addEventListener('canplay', onReady, { once: true });
        }
    } else {
        applyPlayback();
    }
}

function replayActiveShot(targetShot = null, autoScroll = false) {
    const videoElem = document.getElementById('resultVideo');
    if (!videoElem) return;

    if (window.currentActiveShot === 'all' && !targetShot) {
        openFullContinuousVideo(true, autoScroll);
        return;
    }

    const bestNum = (window.currentInsights && window.currentInsights.best_shot) ? window.currentInsights.best_shot : 1;
    let shotNum = targetShot;
    if (!shotNum) {
        if (window.currentActiveShot === 'all' || window.currentActiveShot === 'best') {
            shotNum = bestNum;
        } else {
            shotNum = parseInt(window.currentActiveShot) || 1;
        }
    }

    const timing = getShotVideoTiming(shotNum);
    window.activeShotBoundary = timing;

    videoElem.pause();
    try {
        videoElem.currentTime = timing.start;
    } catch (e) {}
    videoElem.playbackRate = window.currentPlaybackSpeed || 1.0;

    const playPromise = videoElem.play();
    if (playPromise !== undefined) {
        playPromise.then(() => {
            showVideoHudToast(`Playing Shot ${shotNum} (${timing.start.toFixed(1)}s - ${timing.end.toFixed(1)}s) • ${window.currentPlaybackSpeed}x`, '🏏');
        }).catch(e => {
            console.log('Video play deferred, trying muted:', e);
            videoElem.muted = true;
            videoElem.play().catch(() => {});
        });
    }

    updateReplayButtonLabel(shotNum);

    if (autoScroll) {
        const card = document.querySelector('.video-card');
        if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function openFullContinuousVideo(autoPlay = true, autoScroll = true) {
    const dashSec = document.getElementById('shotsDashboardSection');
    const detailSec = document.getElementById('shotDetailSection');
    const navBar = document.getElementById('shotDetailNav');

    if (dashSec) dashSec.classList.add('hidden');
    if (detailSec) detailSec.classList.remove('hidden');

    const shots = window.currentShotsData || [];
    const task = window.currentTaskData || {};
    const tel = task.telemetry || currentTelemetry || {};
    const insights = window.currentInsights || task.insights || {};

    if (navBar && shots.length > 1) {
        navBar.classList.remove('hidden');
        const breadcrumb = document.getElementById('breadcrumbShotTitle');
        if (breadcrumb) breadcrumb.textContent = 'Continuous Full Video';

        // Render Quick Switch pills with Full Video active
        const quickSwitch = document.getElementById('shotQuickSwitch');
        if (quickSwitch) {
            const bestNum = (insights && insights.best_shot) ? insights.best_shot : (task.best_shot || 1);
            let pillsHtml = `
                <button type="button" class="quick-shot-pill active" 
                        onclick="openFullContinuousVideo(true, false)"
                        title="Continuous full video across all deliveries">
                    🎬 Full Video
                </button>
            `;
            pillsHtml += shots.map(s => {
                const isBest = (s.shot === bestNum);
                return `
                    <button type="button" class="quick-shot-pill" 
                            onclick="openShotDetail(${s.shot}, true)"
                            title="Switch directly to Delivery ${s.shot}">
                        ${isBest ? '⭐ ' : ''}Shot ${s.shot}
                    </button>
                `;
            }).join('');
            quickSwitch.innerHTML = pillsHtml;
        }
    }

    // Set active state: all deliveries, no boundary clipping
    window.currentActiveShot = 'all';
    window.activeShotBoundary = null;

    // 1. Update stats cards for full session/spell
    const mainSpd = document.getElementById('statSpeedMain');
    const subSpd = document.getElementById('statSpeedSub');
    const spdSummary = tel.speed_summary || {};
    const peakSpd = spdSummary.peak_speed_kmh || (shots.length ? Math.max(...shots.map(s => s.max_speed_kmh || s.release_speed_kmh || 0)) : null);
    const validAvgShots = shots.filter(s => s.avg_speed_kmh);
    const avgSpd = spdSummary.avg_speed_kmh || (validAvgShots.length ? Math.round(validAvgShots.reduce((acc, s) => acc + s.avg_speed_kmh, 0) / validAvgShots.length) : null);

    if (mainSpd) {
        mainSpd.innerText = peakSpd ? `${peakSpd} km/h` : (avgSpd ? `${avgSpd} km/h` : '-- km/h');
    }
    if (subSpd) {
        subSpd.innerText = `Spell Avg: ${avgSpd || '--'} km/h | Deliveries: ${shots.length || 1} | Best: Shot ${insights.best_shot || task.best_shot || 1}`;
    }

    const rateElem = document.getElementById('statTrackedRate');
    const framesElem = document.getElementById('statTrackedFrames');
    if (rateElem) {
        const rate = (spdSummary.overall_tracking_rate !== undefined) 
            ? spdSummary.overall_tracking_rate 
            : (tel.detected_frames && tel.total_frames ? Math.round((tel.detected_frames / tel.total_frames) * 100) : '--');
        rateElem.innerText = `${rate}%`;
    }
    if (framesElem) {
        framesElem.innerText = `${tel.detected_frames || 0} of ${tel.total_frames || 0} frames`;
    }

    const angleElem = document.getElementById('statAvgAngle');
    if (angleElem) {
        angleElem.innerText = `${task.angle || tel.avg_angle || 0}°`;
    }

    const bounceElem = document.getElementById('statBounces');
    if (bounceElem) {
        const totalBounces = tel.bounce_events ? tel.bounce_events.length : shots.reduce((acc, s) => acc + (s.bounce_count || 0), 0);
        bounceElem.innerText = totalBounces;
    }

    // 2. Update Trim Badge
    const trimBadge = document.getElementById('trimBadge');
    if (trimBadge) {
        const totalDur = (tel.total_frames && tel.fps) 
            ? (tel.total_frames / tel.fps).toFixed(1) 
            : (shots.length && shots[shots.length - 1].end_sec ? shots[shots.length - 1].end_sec.toFixed(1) : '');
        const durStr = totalDur ? ` • ${totalDur}s` : '';
        trimBadge.innerText = `🎬 Continuous Full Video (${shots.length || 1} Deliveries${durStr})`;
        trimBadge.classList.remove('hidden');
    }

    // 3. Update Download Button
    const downloadBtn = document.getElementById('downloadVideoBtn');
    const fullFilename = task.output_filename || ('pred_' + (task.task_id || currentTaskId) + '.mp4');
    if (downloadBtn && fullFilename) {
        downloadBtn.href = `/videos/processed/${fullFilename}`;
        downloadBtn.classList.remove('disabled');
    }

    // 4. Update Replay Button text
    updateReplayButtonLabel('all');

    // 5. Update Pitch Canvas & Insights to all / overlay
    renderPitchShotSelector(shots, 'all');
    if (currentTelemetry) {
        drawPitchTrajectory(currentTelemetry, 'all');
    }
    const bestNum = (insights && insights.best_shot) ? insights.best_shot : (task.best_shot || 1);
    renderShotSelector(shots, insights, 'best');
    renderActiveShotContent('best');

    // 6. Load & Play full continuous video
    const videoElem = document.getElementById('resultVideo');
    if (videoElem && fullFilename) {
        const fullUrl = `/videos/processed/${fullFilename}?t=${task.timestamp || Date.now()}`;
        loadAndPlayVideo(videoElem, fullUrl, 0.0, autoPlay, () => {
            showVideoHudToast(`Playing Continuous Video • ${window.currentPlaybackSpeed || 1.0}x`, '⏮️');
        });
    }

    // 7. Scroll smoothly to video card if requested
    if (autoScroll) {
        const card = document.querySelector('.video-card');
        if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function replayFullVideo() {
    openFullContinuousVideo(true, true);
}

function toggleShotLoop() {
    window.isShotLooping = !window.isShotLooping;
    const loopBtns = document.querySelectorAll('#loopShotToggleBtn, .pitch-loop-btn');
    const loopBtnText = document.getElementById('loopShotBtnText');

    loopBtns.forEach(btn => {
        if (window.isShotLooping) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    if (loopBtnText) {
        loopBtnText.textContent = window.isShotLooping ? 'Loop: ON' : 'Loop: Off';
    }

    if (window.isShotLooping) {
        showVideoHudToast('Shot Repeat Looping: ON', '🔁');
        if (!window.activeShotBoundary) {
            replayActiveShot(null, false);
        }
    } else {
        showVideoHudToast('Shot Repeat Looping: OFF', '⏹️');
    }
}

function updateReplayButtonLabel(shotId) {
    const btnText = document.getElementById('replayShotBtnText');
    if (!btnText) return;
    if (shotId === 'all') {
        btnText.textContent = 'Replay Full Video';
        return;
    }
    const bestNum = (window.currentInsights && window.currentInsights.best_shot) ? window.currentInsights.best_shot : 1;
    const num = (shotId === 'best') ? bestNum : (parseInt(shotId) || 1);
    btnText.textContent = `Replay Shot ${num}`;
}

function initVideoPlaybackListeners() {
    const videoElem = document.getElementById('resultVideo');
    if (!videoElem || videoElem._listenersAttached) return;

    videoElem._listenersAttached = true;

    videoElem.addEventListener('timeupdate', () => {
        if (window.activeShotBoundary) {
            const { start, end, shot } = window.activeShotBoundary;
            if (videoElem.currentTime >= (end - 0.05)) {
                if (window.isShotLooping) {
                    videoElem.currentTime = start;
                    videoElem.play().catch(() => {});
                } else {
                    videoElem.pause();
                    videoElem.currentTime = end;
                    window.activeShotBoundary = null;
                }
            }
        }
    });

    videoElem.addEventListener('ended', () => {
        window.activeShotBoundary = null;
    });
}

function selectShot(shotId, autoPlay = true) {
    window.currentActiveShot = shotId;

    // 1. Update pitch shot selector bar
    renderPitchShotSelector(window.currentShotsData, shotId);

    // 2. Redraw 2D pitch map for this specific shot
    if (currentTelemetry) {
        drawPitchTrajectory(currentTelemetry, shotId);
    }

    // 3. Update AI Insights cards & selector
    const insightShot = (shotId === 'all') ? 'best' : shotId;
    renderShotSelector(window.currentShotsData, window.currentInsights, insightShot);
    renderActiveShotContent(insightShot);

    // 4. Update Replay Button Label & Auto-seek video to this shot's start
    updateReplayButtonLabel(shotId);

    if (shotId !== 'all') {
        const sNum = (shotId === 'best') ? (window.currentInsights?.best_shot || 1) : (parseInt(shotId) || 1);
        const timing = getShotVideoTiming(sNum);
        const videoElem = document.getElementById('resultVideo');
        if (videoElem) {
            window.activeShotBoundary = timing;
            videoElem.currentTime = timing.start;
            videoElem.playbackRate = window.currentPlaybackSpeed || 1.0;
            if (autoPlay) {
                videoElem.play().then(() => {
                    showVideoHudToast(`Shot ${sNum} (${timing.start.toFixed(1)}s - ${timing.end.toFixed(1)}s) • ${window.currentPlaybackSpeed}x`, '🏏');
                }).catch(() => {});
            }
        }
    } else {
        window.activeShotBoundary = null;
        const videoElem = document.getElementById('resultVideo');
        if (videoElem) {
            videoElem.currentTime = 0.0;
            videoElem.playbackRate = window.currentPlaybackSpeed || 1.0;
            if (autoPlay) {
                videoElem.play().then(() => {
                    showVideoHudToast(`Playing Continuous Video • ${window.currentPlaybackSpeed}x`, '⏮️');
                }).catch(err => {
                    videoElem.muted = true;
                    videoElem.play().catch(() => {});
                });
            } else {
                showVideoHudToast('Overlay View: All Deliveries Mapped', '🌐');
            }
        }
    }
}

function renderPitchShotSelector(shotsData, activeShot) {
    const bar = document.getElementById('pitchShotSelectorBar');
    if (!bar) return;

    if (!shotsData || shotsData.length <= 1) {
        bar.innerHTML = '';
        bar.classList.add('hidden');
        return;
    }

    bar.classList.remove('hidden');

    const bestNum = (window.currentInsights && window.currentInsights.best_shot) ? window.currentInsights.best_shot : 1;
    const currentShotNum = (activeShot === 'all')
        ? 1
        : ((activeShot === 'best') ? bestNum : (parseInt(activeShot) || 1));
    const curr = shotsData.find(s => s.shot === currentShotNum) || shotsData[0] || {};
    const currentTiming = getShotVideoTiming(currentShotNum);

    let pillsHtml = shotsData.map(s => {
        const isBest = (s.shot === bestNum);
        const isActive = (activeShot === s.shot || String(activeShot) === String(s.shot) || (activeShot === 'best' && isBest));
        return `
            <button type="button" class="pitch-shot-btn ${isActive ? 'active' : ''} ${isBest ? 'is-best' : ''}" onclick="selectShot(${s.shot}, true)" title="Inspect & Play Shot ${s.shot} in Video">
                <span class="pitch-shot-indicator shot-${s.shot}"></span>
                <span>Shot ${s.shot}</span>
                ${isBest ? '<span class="pitch-best-badge">⭐ BEST</span>' : ''}
                <span class="pitch-shot-speed">${s.release_speed_kmh ? s.release_speed_kmh + ' km/h' : ''}</span>
            </button>
        `;
    }).join('');

    pillsHtml += `
        <button type="button" class="pitch-shot-btn ${activeShot === 'all' ? 'active' : ''}" onclick="selectShot('all', false)" title="Overlay all deliveries on one map">
            <span>🌐 Overlay All</span>
        </button>
    `;

    const infoText = (activeShot === 'all')
        ? `Viewing Multi-Delivery Overlay (${shotsData.length} Shots)`
        : `Shot ${curr.shot} of ${shotsData.length} (${currentTiming.start.toFixed(1)}s - ${currentTiming.end.toFixed(1)}s) • Release: ${curr.release_speed_kmh || '--'} km/h • Bounces: ${curr.bounce_count ?? (curr.bounces ? curr.bounces.length : 0)} • Angle: ${curr.avg_angle || 0}°`;

    bar.innerHTML = `
        <div class="pitch-shot-pills">${pillsHtml}</div>
        <div class="pitch-shot-actions">
            <button type="button" class="pitch-replay-btn" onclick="replayActiveShot(${currentShotNum}, true)" title="Replay Shot ${currentShotNum} in Video Player">
                <span>▶️ Replay Shot ${currentShotNum}</span>
            </button>
            <button type="button" class="ctrl-pill-btn btn-loop-shot pitch-loop-btn ${window.isShotLooping ? 'active' : ''}" onclick="toggleShotLoop()" title="Toggle repeat looping for this shot" style="padding: 0.35rem 0.7rem; font-size: 0.74rem;">
                <span>🔁 ${window.isShotLooping ? 'Loop: ON' : 'Loop'}</span>
            </button>
            <button type="button" class="ctrl-pill-btn btn-replay-all" onclick="replayFullVideo()" title="Replay full video from start" style="padding: 0.35rem 0.7rem; font-size: 0.74rem;">
                <span>⏮️ Replay Full</span>
            </button>
            <div class="pitch-speed-pills">
                <button type="button" class="pitch-speed-pill ${window.currentPlaybackSpeed === 0.25 ? 'active' : ''}" data-speed="0.25" onclick="setVideoSpeed(0.25)" title="0.25x Super Slow Motion">0.25x</button>
                <button type="button" class="pitch-speed-pill ${window.currentPlaybackSpeed === 0.5 ? 'active' : ''}" data-speed="0.5" onclick="setVideoSpeed(0.5)" title="0.5x Slow Motion">0.5x</button>
                <button type="button" class="pitch-speed-pill ${window.currentPlaybackSpeed === 1.0 ? 'active' : ''}" data-speed="1.0" onclick="setVideoSpeed(1.0)" title="1x Normal Speed">1x</button>
            </div>
            <div class="pitch-shot-info-pill">${infoText}</div>
        </div>
    `;
}

function renderShotSelector(shotsData, insights, activeShot) {
    const container = document.getElementById('shotSelectorContainer');
    if (!container) return;

    if (!shotsData || shotsData.length <= 1) {
        container.innerHTML = '';
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');

    const bestShotNum = (insights && typeof insights === 'object') ? (insights.best_shot || null) : null;

    let html = `
        <div class="shot-selector-label">
            <span class="shot-selector-title">Select Delivery Shot:</span>
            <span class="shot-selector-hint">${shotsData.length} Deliveries Segmented</span>
        </div>
        <div class="shot-pill-group">
            <button type="button" class="shot-pill ${activeShot === 'best' ? 'active' : ''}" onclick="selectShot('best')">
                <span class="pill-icon">🏆</span>
                <span class="pill-text">Best Shot Verdict</span>
            </button>
    `;

    shotsData.forEach(s => {
        const isBest = (bestShotNum && s.shot === bestShotNum);
        const isActive = (activeShot === s.shot || activeShot === String(s.shot));
        html += `
            <button type="button" class="shot-pill ${isActive ? 'active' : ''} ${isBest ? 'winner-pill' : ''}" onclick="selectShot(${s.shot})">
                <span class="pill-dot shot-${s.shot}"></span>
                <span class="pill-text">Shot ${s.shot}</span>
                ${isBest ? '<span class="pill-badge">⭐ BEST</span>' : ''}
                <span class="pill-speed">${s.release_speed_kmh ? s.release_speed_kmh + ' km/h' : ''}</span>
            </button>
        `;
    });

    html += `</div>`;
    container.innerHTML = html;
}

function renderActiveShotContent(shotId) {
    const body = document.getElementById('aiInsightsBody');
    if (!body) return;

    const shotsData = window.currentShotsData || [];
    const insights = window.currentInsights;
    const tel = currentTelemetry || {};
    const spd = tel.speed_summary || {};
    const bounces = tel.bounce_events || [];
    const trim = tel.trim_info || {};

    // 1. Build Stat Chips based on shot selection
    let chips = [];
    if (shotId === 'best' || shotsData.length === 0) {
        // Global stats
        chips = [
            { label: 'Release', value: spd.release_speed_kmh ? `${spd.release_speed_kmh} km/h` : '--' },
            { label: 'Avg Speed', value: spd.avg_speed_kmh ? `${spd.avg_speed_kmh} km/h` : '--' },
            { label: 'Peak', value: spd.max_speed_kmh ? `${spd.max_speed_kmh} km/h` : '--' },
            { label: 'Total Bounces', value: bounces.length },
            { label: 'Total Shots', value: shotsData.length || (trim.total_shots ?? 1) }
        ];
    } else {
        // Per-shot stats
        const sNum = parseInt(shotId);
        const s = shotsData.find(x => x.shot === sNum) || {};
        chips = [
            { label: `Shot ${sNum} Release`, value: s.release_speed_kmh ? `${s.release_speed_kmh} km/h` : '--' },
            { label: 'Avg Speed', value: s.avg_speed_kmh ? `${s.avg_speed_kmh} km/h` : '--' },
            { label: 'Peak Speed', value: s.max_speed_kmh ? `${s.max_speed_kmh} km/h` : '--' },
            { label: 'Bounces', value: s.bounce_count ?? (s.bounces ? s.bounces.length : 0) },
            { label: 'Trajectory Angle', value: s.avg_angle ? `${s.avg_angle}°` : '--' },
            { label: 'Duration', value: s.duration_sec ? `${s.duration_sec}s` : '--' }
        ];
    }

    const chipHtml = chips.map(c =>
        `<div class="ai-stat-chip"><span>${c.label}</span>${c.value}</div>`
    ).join('');

    // 2. Build insights prose / cards
    let contentHtml = '';

    if (!insights) {
        // No insights yet
        contentHtml = `
            <div class="ai-empty-state">
                <div class="ai-empty-icon">🏏</div>
                <div>${shotsData.length > 1 ? `Found <strong>${shotsData.length} deliveries</strong> in this video. Click below to generate shot-wise analysis and find out which shot was best!` : 'Click <strong>Get AI Insights</strong> to get Gemini-powered coaching analysis.'}</div>
                <button class="btn-small btn-ai" style="margin-top:0.85rem;" onclick="runAIAnalysis()">✨ Generate Shot-Wise Coaching Insights</button>
            </div>
        `;
    } else if (typeof insights === 'object' && insights.shots) {
        // Structured Gemini Insights
        if (shotId === 'best') {
            // Comparative Verdict Overview
            const bestNum = insights.best_shot || 1;
            contentHtml = `
                <div class="best-shot-banner">
                    <div class="best-shot-badge-top">🏆 BEST SHOT VERDICT</div>
                    <h3 class="best-shot-headline">${escapeHtml(insights.best_shot_title || ('Shot ' + bestNum + ' was the Standout Delivery'))}</h3>
                    <p class="best-shot-verdict-text">${escapeHtml(insights.verdict || '')}</p>
                </div>
                <div class="shots-summary-headline">All Deliveries Comparison:</div>
                <div class="shots-compare-grid">
                    ${shotsData.map(s => {
                        const isWinner = (s.shot === bestNum);
                        const sInsight = insights.shots.find(x => x.shot === s.shot);
                        return `
                            <div class="shot-mini-card ${isWinner ? 'winner-card' : ''}" onclick="selectShot(${s.shot})">
                                <div class="mini-card-head">
                                    <div class="mini-card-shot-num">Shot ${s.shot}</div>
                                    ${isWinner ? '<span class="badge badge-winner">⭐ Best Shot</span>' : ''}
                                    <span class="badge badge-threat">${escapeHtml(sInsight?.threat_rating || '7/10')}</span>
                                </div>
                                <div class="mini-card-type">${escapeHtml(sInsight?.delivery_type || 'Delivery ' + s.shot)}</div>
                                <div class="mini-card-metrics">
                                    <div class="mini-metric"><span>Release:</span> <strong>${s.release_speed_kmh} km/h</strong></div>
                                    <div class="mini-metric"><span>Peak:</span> <strong>${s.max_speed_kmh} km/h</strong></div>
                                    <div class="mini-metric"><span>Bounces:</span> <strong>${s.bounce_count}</strong></div>
                                </div>
                                <div class="mini-card-action">View Shot ${s.shot} Breakdown →</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        } else {
            // Individual Shot Breakdown
            const sNum = parseInt(shotId);
            const shotInsight = insights.shots.find(x => x.shot === sNum) || {};
            const isWinner = (insights.best_shot === sNum);

            contentHtml = `
                <div class="single-shot-panel">
                    <div class="shot-detail-header">
                        <div class="shot-detail-badges">
                            <span class="badge badge-shot">Shot ${sNum} of ${shotsData.length}</span>
                            ${isWinner ? '<span class="badge badge-winner">⭐ Standout Delivery (Best Shot)</span>' : ''}
                            <span class="badge badge-threat">Threat: ${escapeHtml(shotInsight.threat_rating || '7/10')}</span>
                        </div>
                        <h3 class="shot-detail-title">${escapeHtml(shotInsight.title || ('Shot ' + sNum + ': ' + (shotInsight.delivery_type || 'Delivery Analysis')))}</h3>
                        ${shotInsight.summary ? `<p class="shot-detail-summary">${escapeHtml(shotInsight.summary)}</p>` : ''}
                    </div>

                    <div class="shot-bullet-list">
                        ${(shotInsight.points || []).map(pt => `
                            <div class="shot-bullet-item">
                                <span class="bullet-icon">⚡</span>
                                <div class="bullet-text">${simpleMarkdown(pt)}</div>
                            </div>
                        `).join('')}
                    </div>

                    <div class="shot-switch-hint">
                        <button type="button" class="btn-subtle" onclick="selectShot('best')">← Back to Best Shot Comparison</button>
                    </div>
                </div>
            `;
        }
    } else {
        // Raw Markdown string fallback (e.g. older tasks before JSON structured schema)
        contentHtml = `
            <div class="legacy-insights-banner">
                <span>Displaying delivery report.</span>
                <button class="btn-small btn-ai" style="margin-left: auto;" onclick="runAIAnalysis()">⚡ Convert to Shot-Wise Insights</button>
            </div>
            <div class="ai-prose">${simpleMarkdown(String(insights))}</div>
        `;
    }

    body.innerHTML = `
        <div class="ai-stats-row">${chipHtml}</div>
        <div class="ai-insights-content">${contentHtml}</div>
    `;
}

function displayResults(task) {
    currentTelemetry = task.telemetry;
    lastCompletedTaskId = task.task_id || currentTaskId;
    window.currentTaskData = task;

    const tel = task.telemetry || {};
    window.currentShotsData = (tel.shots_data && tel.shots_data.length > 0) ? tel.shots_data : [];
    window.currentInsights = task.insights || null;

    // Enable AI Insights button
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        if (task.insights) {
            analyzeBtn.textContent = '🔄 Re-analyze Insights';
        } else {
            analyzeBtn.textContent = '✨ Get AI Insights';
        }
    }

    const isMultiShot = window.currentShotsData.length > 1;

    if (isMultiShot) {
        // Multi-Shot Workflow: Show Dashboard Grid first
        renderShotsDashboard(task, window.currentShotsData);
        const dashSec = document.getElementById('shotsDashboardSection');
        const detailSec = document.getElementById('shotDetailSection');
        if (dashSec) dashSec.classList.remove('hidden');
        if (detailSec) detailSec.classList.add('hidden');

        // Pre-initialize selectors for detail view
        const bestNum = (window.currentInsights && window.currentInsights.best_shot) ? window.currentInsights.best_shot : (task.best_shot || 1);
        renderPitchShotSelector(window.currentShotsData, bestNum);
        renderShotSelector(window.currentShotsData, window.currentInsights, 'best');
        renderActiveShotContent('best');

        // Render Telemetry Table & Bounce Log for full session
        renderTelemetryTable(tel.frame_data || []);
        renderBounceLog(tel.bounce_events || []);
    } else {
        // Single delivery (N=1): Directly open detail view
        const dashSec = document.getElementById('shotsDashboardSection');
        const detailSec = document.getElementById('shotDetailSection');
        const navBar = document.getElementById('shotDetailNav');
        if (dashSec) dashSec.classList.add('hidden');
        if (detailSec) detailSec.classList.remove('hidden');
        if (navBar) navBar.classList.add('hidden');

        openShotDetail(1, false);

        // Render Telemetry Table & Bounce Log
        renderTelemetryTable(tel.frame_data || []);
        renderBounceLog(tel.bounce_events || []);
    }

    // Refresh history catalog
    loadHistory();
}

function renderShotsDashboard(task, shotsData) {
    const container = document.getElementById('shotsGridContainer');
    const badge = document.getElementById('dashSpellBadge');
    const summaryElem = document.getElementById('dashSpellSummary');

    if (!container || !shotsData || shotsData.length === 0) return;

    if (badge) {
        badge.textContent = `${shotsData.length} Deliveries`;
    }

    // Find standout / best shot
    const bestNum = (window.currentInsights && window.currentInsights.best_shot) 
        ? window.currentInsights.best_shot 
        : (task.best_shot || 1);

    // Calculate maximum release speed
    let maxRelease = 0;
    shotsData.forEach(s => {
        const rel = parseFloat(s.release_speed_kmh || 0);
        if (rel > maxRelease) maxRelease = rel;
    });

    if (summaryElem) {
        summaryElem.innerHTML = `
            <strong>${shotsData.length} auto-segmented deliveries</strong> • Peak release: <strong>${maxRelease > 0 ? maxRelease + ' km/h' : '--'}</strong> • 
            Standout: <span style="color:#eeb20d; font-weight:800;">Shot ${bestNum}</span>. Click any delivery card below to inspect telemetry & video.
        `;
    }

    container.innerHTML = shotsData.map(s => {
        const isBest = (s.shot === bestNum || s.is_best === true);
        const taskId = task.task_id || currentTaskId || '';
        const thumbSrc = s.thumbnail_url || (taskId ? `/videos/processed/${taskId}/thumb_shot_${s.shot}.jpg` : '');
        const relKmh = s.release_speed_kmh || s.avg_speed_kmh || '--';
        const maxKmh = s.max_speed_kmh || '--';
        const dur = s.duration_sec || '--';
        const bounces = s.bounce_count !== undefined ? s.bounce_count : (s.bounces ? s.bounces.length : 0);
        const angle = s.avg_angle !== undefined ? `${s.avg_angle}°` : '--°';

        // Check if Gemini insights exist for this shot
        let typeTag = 'Delivery';
        let summaryText = `Clocked at ${relKmh} km/h release reaching ${maxKmh} km/h peak over ${dur}s.`;
        if (window.currentInsights && window.currentInsights.shots) {
            const ins = window.currentInsights.shots.find(item => item.shot === s.shot);
            if (ins) {
                if (ins.delivery_type) typeTag = ins.delivery_type;
                if (ins.summary) summaryText = ins.summary;
            }
        }

        return `
            <div class="shot-card ${isBest ? 'winner-card' : ''}" onclick="openShotDetail(${s.shot}, true)">
                <div class="shot-card-thumb-wrap">
                    ${thumbSrc ? `
                        <img src="${thumbSrc}" alt="Shot ${s.shot}" class="shot-card-thumb" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                        <div class="shot-card-thumb-placeholder" style="display:none;">
                            <span style="font-size:2rem;">🏏</span>
                            <span>Shot ${s.shot} Clip</span>
                        </div>
                    ` : `
                        <div class="shot-card-thumb-placeholder">
                            <span style="font-size:2rem;">🏏</span>
                            <span>Shot ${s.shot} Clip</span>
                        </div>
                    `}
                    <div class="shot-thumb-badge-left">Shot ${s.shot}</div>
                    <div class="shot-thumb-badge-right">⏱️ ${dur}s</div>
                    ${isBest ? `
                        <div class="shot-winner-ribbon">
                            <span>⭐</span>
                            <span>Standout Delivery</span>
                        </div>
                    ` : ''}
                </div>

                <div class="shot-card-body">
                    <div class="shot-card-header">
                        <h3 class="shot-card-title">Delivery #${s.shot}</h3>
                        <span class="shot-card-type-tag">${escapeHtml(typeTag)}</span>
                    </div>

                    <div class="shot-metrics-grid">
                        <div class="shot-metric-box ${isBest ? 'highlight' : ''}">
                            <span class="shot-metric-label">Release Speed</span>
                            <span class="shot-metric-val gold">${relKmh} km/h</span>
                        </div>
                        <div class="shot-metric-box">
                            <span class="shot-metric-label">Peak Speed</span>
                            <span class="shot-metric-val cyan">${maxKmh} km/h</span>
                        </div>
                        <div class="shot-metric-box">
                            <span class="shot-metric-label">Pitch Bounces</span>
                            <span class="shot-metric-val green">${bounces} event${bounces === 1 ? '' : 's'}</span>
                        </div>
                        <div class="shot-metric-box">
                            <span class="shot-metric-label">Angle</span>
                            <span class="shot-metric-val">${angle}</span>
                        </div>
                    </div>

                    <p class="shot-card-summary">${escapeHtml(summaryText)}</p>

                    <button type="button" class="shot-card-btn" onclick="event.stopPropagation(); openShotDetail(${s.shot}, true)">
                        <span>Inspect Delivery ${s.shot}</span>
                        <span>→</span>
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function openShotDetail(shotNumber, autoPlay = true) {
    const dashSec = document.getElementById('shotsDashboardSection');
    const detailSec = document.getElementById('shotDetailSection');
    const navBar = document.getElementById('shotDetailNav');

    if (dashSec) dashSec.classList.add('hidden');
    if (detailSec) detailSec.classList.remove('hidden');

    const shots = window.currentShotsData || [];
    const isMultiShot = shots.length > 1;

    // Update navigation breadcrumb and quick switches if multi-shot
    if (navBar) {
        if (isMultiShot) {
            navBar.classList.remove('hidden');
            const breadcrumb = document.getElementById('breadcrumbShotTitle');
            if (breadcrumb) {
                breadcrumb.textContent = `Delivery ${shotNumber} Detail`;
            }

            // Quick switch pills (includes Full Video button!)
            const quickSwitch = document.getElementById('shotQuickSwitch');
            if (quickSwitch) {
                const bestNum = (window.currentInsights && window.currentInsights.best_shot) 
                    ? window.currentInsights.best_shot 
                    : (window.currentTaskData?.best_shot || 1);
                let pillsHtml = `
                    <button type="button" class="quick-shot-pill ${shotNumber === 'all' ? 'active' : ''}" 
                            onclick="openFullContinuousVideo(true, false)"
                            title="Watch continuous full video across all deliveries">
                        🎬 Full Video
                    </button>
                `;
                pillsHtml += shots.map(s => {
                    const isActive = (s.shot === shotNumber);
                    const isBest = (s.shot === bestNum);
                    return `
                        <button type="button" class="quick-shot-pill ${isActive ? 'active' : ''}" 
                                onclick="openShotDetail(${s.shot}, true)"
                                title="Switch directly to Delivery ${s.shot}">
                            ${isBest ? '⭐ ' : ''}Shot ${s.shot}
                        </button>
                    `;
                }).join('');
                quickSwitch.innerHTML = pillsHtml;
            }
        } else {
            navBar.classList.add('hidden');
        }
    }

    window.currentActiveShot = shotNumber;

    // Find current shot object
    const shot = shots.find(s => s.shot === shotNumber) || shots[0] || {};
    const task = window.currentTaskData || {};

    // 1. Update stats cards for this specific delivery
    if (shot) {
        const mainSpd = document.getElementById('statSpeedMain');
        const subSpd = document.getElementById('statSpeedSub');
        if (mainSpd) {
            const relVal = shot.release_speed_kmh || shot.avg_speed_kmh;
            mainSpd.innerText = relVal ? `${relVal} km/h` : '-- km/h';
        }
        if (subSpd) {
            subSpd.innerText = `Avg: ${shot.avg_speed_kmh || '--'} km/h | Peak: ${shot.max_speed_kmh || '--'} km/h`;
        }

        const rateElem = document.getElementById('statTrackedRate');
        const framesElem = document.getElementById('statTrackedFrames');
        if (rateElem && shot.tracking_rate !== undefined) {
            rateElem.innerText = `${shot.tracking_rate}%`;
        }
        if (framesElem && shot.total_frames !== undefined) {
            framesElem.innerText = `${shot.detected_frames || 0} of ${shot.total_frames} frames`;
        }

        const angleElem = document.getElementById('statAvgAngle');
        if (angleElem && shot.avg_angle !== undefined) {
            angleElem.innerText = `${shot.avg_angle}°`;
        }

        const bounceElem = document.getElementById('statBounces');
        if (bounceElem && shot.bounce_count !== undefined) {
            bounceElem.innerText = shot.bounce_count;
        }

        // Update Trim Badge
        const trimBadge = document.getElementById('trimBadge');
        if (trimBadge) {
            trimBadge.innerText = `✂️ Delivery ${shotNumber}: ${shot.start_sec || 0}s ➔ ${shot.end_sec || 0}s (${shot.duration_sec || 0}s)`;
            trimBadge.classList.remove('hidden');
        }
    }

    // 2. Load Video - prefer dedicated per-shot video clip if available
    const videoElem = document.getElementById('resultVideo');
    const downloadBtn = document.getElementById('downloadVideoBtn');

    let targetVideoUrl = null;
    let isIndependentClip = false;

    if (shot.video_url) {
        targetVideoUrl = shot.video_url;
        isIndependentClip = true;
    } else if (task.output_filename) {
        targetVideoUrl = `/videos/processed/${task.output_filename}`;
    }

    if (videoElem && targetVideoUrl) {
        const fullTargetUrl = `${targetVideoUrl}?t=${task.timestamp || Date.now()}`;
        if (isIndependentClip) {
            window.activeShotBoundary = null;
            loadAndPlayVideo(videoElem, fullTargetUrl, 0.0, autoPlay, () => {
                showVideoHudToast(`Shot ${shotNumber} Clip • ${window.currentPlaybackSpeed || 1.0}x`, '🏏');
            });
        } else {
            const timing = getShotVideoTiming(shotNumber);
            window.activeShotBoundary = timing;
            loadAndPlayVideo(videoElem, fullTargetUrl, timing.start, autoPlay, () => {
                showVideoHudToast(`Shot ${shotNumber} (${timing.start.toFixed(1)}s - ${timing.end.toFixed(1)}s) • ${window.currentPlaybackSpeed || 1.0}x`, '🏏');
            });
        }
    }

    if (downloadBtn) {
        if (shot.video_url) {
            downloadBtn.href = shot.video_url;
        } else if (task.output_filename) {
            downloadBtn.href = `/videos/processed/${task.output_filename}`;
        }
        downloadBtn.classList.remove('disabled');
    }

    // 3. Select active shot for pitch canvas and AI insights
    selectShot(shotNumber, false);

    // 4. Scroll smoothly to detail view if in multi-shot
    if (autoPlay && isMultiShot) {
        detailSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function backToShotsDashboard() {
    const dashSec = document.getElementById('shotsDashboardSection');
    const detailSec = document.getElementById('shotDetailSection');
    const videoElem = document.getElementById('resultVideo');

    if (videoElem) {
        videoElem.pause();
    }

    if (detailSec) detailSec.classList.add('hidden');
    if (dashSec) {
        dashSec.classList.remove('hidden');
        dashSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function initPitchCanvas() {
    const canvas = document.getElementById('pitchCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';

    // Background Turf
    ctx.fillStyle = isLight ? '#E1EDE6' : '#07170f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Wicket pitch strip
    const pitchWidth = 620;
    const pitchHeight = 150;
    const pitchX = (canvas.width - pitchWidth) / 2;
    const pitchY = (canvas.height - pitchHeight) / 2;

    // Pitch surface
    ctx.fillStyle = isLight ? '#D0E0D6' : '#14271c';
    ctx.fillRect(pitchX, pitchY, pitchWidth, pitchHeight);
    ctx.strokeStyle = isLight ? '#A5BEAF' : '#223f2f';
    ctx.lineWidth = 2;
    ctx.strokeRect(pitchX, pitchY, pitchWidth, pitchHeight);

    // Good Length Channel Guide (subtle dashed box)
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pitchX + 340, pitchY + 6, 150, pitchHeight - 12);
    ctx.font = '600 9px Inter, sans-serif';
    ctx.fillStyle = isLight ? 'rgba(0, 0, 0, 0.25)' : 'rgba(255, 255, 255, 0.22)';
    ctx.fillText('GOOD LENGTH ZONE', pitchX + 365, pitchY + pitchHeight - 12);
    ctx.restore();

    // Bowling & Popping Creases
    ctx.strokeStyle = isLight ? 'rgba(25, 55, 35, 0.75)' : 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 2;

    // Bowler end crease (Left)
    ctx.beginPath();
    ctx.moveTo(pitchX + 60, pitchY);
    ctx.lineTo(pitchX + 60, pitchY + pitchHeight);
    ctx.stroke();

    // Bowler return creases
    ctx.beginPath();
    ctx.moveTo(pitchX + 30, pitchY);
    ctx.lineTo(pitchX + 60, pitchY);
    ctx.moveTo(pitchX + 30, pitchY + pitchHeight);
    ctx.lineTo(pitchX + 60, pitchY + pitchHeight);
    ctx.stroke();

    // Batsman end popping crease (Right)
    ctx.beginPath();
    ctx.moveTo(pitchX + pitchWidth - 60, pitchY);
    ctx.lineTo(pitchX + pitchWidth - 60, pitchY + pitchHeight);
    ctx.stroke();

    // Batsman return creases
    ctx.beginPath();
    ctx.moveTo(pitchX + pitchWidth - 60, pitchY);
    ctx.lineTo(pitchX + pitchWidth - 30, pitchY);
    ctx.moveTo(pitchX + pitchWidth - 60, pitchY + pitchHeight);
    ctx.lineTo(pitchX + pitchWidth - 30, pitchY + pitchHeight);
    ctx.stroke();

    // Stumps (Orange dots)
    ctx.fillStyle = isLight ? '#D97706' : '#F59E0B';
    // Bowler stumps (3 dots)
    const bStumpX = pitchX + 40;
    const midY = pitchY + pitchHeight / 2;
    [-7, 0, 7].forEach(offset => {
        ctx.beginPath();
        ctx.arc(bStumpX, midY + offset, 2.5, 0, Math.PI * 2);
        ctx.fill();
    });

    // Batsman stumps (3 dots)
    const batStumpX = pitchX + pitchWidth - 40;
    [-7, 0, 7].forEach(offset => {
        ctx.beginPath();
        ctx.arc(batStumpX, midY + offset, 2.5, 0, Math.PI * 2);
        ctx.fill();
    });

    // Labels
    ctx.fillStyle = isLight ? 'rgba(25, 55, 35, 0.85)' : 'rgba(255, 255, 255, 0.65)';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText('BOWLER END', pitchX + 15, pitchY - 10);
    ctx.fillText('BATSMAN END (STUMPS)', pitchX + pitchWidth - 150, pitchY - 10);
}

function drawPitchTrajectory(telemetry, activeShot = 1) {
    initPitchCanvas();
    const canvas = document.getElementById('pitchCanvas');
    if (!canvas || !telemetry) return;
    const ctx = canvas.getContext('2d');

    const points = telemetry.trajectory_points;
    if (!points || points.length === 0) return;

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const w = telemetry.width || 640;
    const h = telemetry.height || 360;

    const pitchWidth = 620;
    const pitchHeight = 150;
    const pitchX = (canvas.width - pitchWidth) / 2;
    const pitchY = (canvas.height - pitchHeight) / 2;

    const creaseStart = pitchX + 65;
    const creaseEnd = pitchX + pitchWidth - 65;
    const playLength = creaseEnd - creaseStart;
    const midPitchY = pitchY + pitchHeight / 2;

    // Distinct vibrant color palettes per shot
    const shotColors = [
        { stroke: '#38bdf8', glow: 'rgba(56, 189, 248, 0.45)', dot: '#0ea5e9', lastDot: '#f43f5e', name: 'Shot 1' },
        { stroke: '#10b981', glow: 'rgba(16, 185, 129, 0.45)', dot: '#059669', lastDot: '#f59e0b', name: 'Shot 2' },
        { stroke: '#f59e0b', glow: 'rgba(245, 158, 11, 0.45)', dot: '#d97706', lastDot: '#ec4899', name: 'Shot 3' },
        { stroke: '#a855f7', glow: 'rgba(168, 85, 247, 0.45)', dot: '#7e22ce', lastDot: '#ef4444', name: 'Shot 4' },
        { stroke: '#ec4899', glow: 'rgba(236, 72, 153, 0.45)', dot: '#db2777', lastDot: '#3b82f6', name: 'Shot 5' }
    ];

    // Determine target shot number
    const totalShots = (telemetry.shots_data && telemetry.shots_data.length > 0)
        ? telemetry.shots_data.length
        : (telemetry.total_shots || 1);

    let targetShotNum = null;
    if (activeShot === 'best') {
        targetShotNum = (window.currentInsights && window.currentInsights.best_shot)
            ? window.currentInsights.best_shot
            : 1;
    } else if (activeShot !== 'all') {
        targetShotNum = parseInt(activeShot) || 1;
    }

    // Group trajectory points by shot
    const shotsMap = {};
    points.forEach(pt => {
        const s = pt.shot || 1;
        if (!shotsMap[s]) shotsMap[s] = [];
        shotsMap[s].push(pt);
    });

    // Determine which shots to render:
    // When viewing a specific shot, strictly render ONLY that shot — NO ghost lines from other shots!
    let shotsToRender = [];
    if (targetShotNum !== null) {
        if (shotsMap[targetShotNum]) {
            shotsToRender = [targetShotNum];
        } else {
            const firstAvailable = Object.keys(shotsMap)[0];
            if (firstAvailable) shotsToRender = [parseInt(firstAvailable)];
        }
    } else {
        shotsToRender = Object.keys(shotsMap).map(k => parseInt(k)).sort((a, b) => a - b);
    }

    // 1. Draw Trajectory Line & Nodes for active shots
    shotsToRender.forEach(sNum => {
        const sPoints = shotsMap[sNum];
        if (!sPoints || sPoints.length === 0) return;

        sPoints.sort((a, b) => a.frame - b.frame);
        const fStart = sPoints[0].frame;
        const fSpan = Math.max(1, sPoints[sPoints.length - 1].frame - fStart);
        const color = shotColors[(sNum - 1) % shotColors.length];
        const currShotInfo = (telemetry.shots_data || []).find(s => s.shot === sNum) || {};

        const mapped = sPoints.map(pt => {
            const prog = (fSpan > 0) ? ((pt.frame - fStart) / fSpan) : 0.5;
            const cx = creaseStart + Math.max(0, Math.min(1, prog)) * playLength;
            const latNorm = (pt.x - (w * 0.5)) / (w * 0.5);
            const cy = midPitchY + Math.max(-55, Math.min(55, latNorm * 45));
            return { x: cx, y: cy, frame: pt.frame, orig: pt };
        });

        if (mapped.length > 1) {
            // Check if points contain phase information (Two-Phase Trajectory)
            const hasPhases = mapped.some(m => m.orig && m.orig.phase);
            const p1Mapped = mapped.filter(m => !m.orig || m.orig.phase === 1 || !m.orig.phase);
            const p2Mapped = mapped.filter(m => m.orig && m.orig.phase === 2);

            if (hasPhases && p2Mapped.length > 0) {
                // Ensure p2Mapped connects cleanly to the bounce point
                if (p1Mapped.length > 0 && p2Mapped[0] !== p1Mapped[p1Mapped.length - 1]) {
                    p2Mapped.unshift(p1Mapped[p1Mapped.length - 1]);
                }

                // Phase 1 (Pre-bounce flight: Amber / Gold)
                if (p1Mapped.length > 1) {
                    ctx.save();
                    ctx.strokeStyle = '#f59e0b';
                    ctx.lineWidth = (targetShotNum !== null) ? 4 : 3;
                    ctx.shadowColor = 'rgba(245, 158, 11, 0.45)';
                    ctx.shadowBlur = 10;
                    ctx.beginPath();
                    ctx.moveTo(p1Mapped[0].x, p1Mapped[0].y);
                    for (let i = 1; i < p1Mapped.length; i++) {
                        ctx.lineTo(p1Mapped[i].x, p1Mapped[i].y);
                    }
                    ctx.stroke();
                    ctx.restore();
                }

                // Phase 2 (Post-bounce flight to stumps: Electric Cyan)
                if (p2Mapped.length > 1) {
                    ctx.save();
                    ctx.strokeStyle = '#00e5ff';
                    ctx.lineWidth = (targetShotNum !== null) ? 4 : 3;
                    ctx.shadowColor = 'rgba(0, 229, 255, 0.5)';
                    ctx.shadowBlur = 12;
                    ctx.beginPath();
                    ctx.moveTo(p2Mapped[0].x, p2Mapped[0].y);
                    for (let i = 1; i < p2Mapped.length; i++) {
                        ctx.lineTo(p2Mapped[i].x, p2Mapped[i].y);
                    }
                    ctx.stroke();
                    ctx.restore();
                }
            } else {
                // Standard continuous stroke
                ctx.save();
                ctx.strokeStyle = color.stroke;
                ctx.lineWidth = (targetShotNum !== null) ? 4 : 3;
                ctx.shadowColor = color.glow;
                ctx.shadowBlur = 10;
                ctx.beginPath();
                ctx.moveTo(mapped[0].x, mapped[0].y);
                for (let i = 1; i < mapped.length; i++) {
                    ctx.lineTo(mapped[i].x, mapped[i].y);
                }
                ctx.stroke();
                ctx.restore();
            }

            // Draw trajectory point nodes
            mapped.forEach((pt, idx) => {
                const isFirst = (idx === 0);
                const isLast = (idx === mapped.length - 1);
                const isPhase2 = pt.orig && pt.orig.phase === 2;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, (isFirst || isLast) ? 5.5 : 3.5, 0, Math.PI * 2);
                ctx.fillStyle = isLast ? color.lastDot : (isFirst ? '#ffffff' : (isPhase2 ? '#00e5ff' : color.dot));
                ctx.fill();
                if (isFirst) {
                    ctx.strokeStyle = color.stroke;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            });

            // Release callout at bowler end
            const rPt = mapped[0];
            const relSpd = currShotInfo.release_speed_kmh || telemetry.speed_summary?.release_speed_kmh;
            const relLabel = relSpd ? `Release ${relSpd} km/h` : 'Release';
            ctx.font = '600 10px Inter, sans-serif';
            ctx.fillStyle = isLight ? '#1e293b' : 'rgba(255, 255, 255, 0.85)';
            ctx.fillText(relLabel, Math.max(pitchX + 10, rPt.x - 18), rPt.y - 12);

            // Stumps Hit / Impact Badge at Batsman End
            if (currShotInfo.is_stump_hit || currShotInfo.stump_hit_point) {
                const sPt = mapped[mapped.length - 1];
                const hitX = Math.min(pitchX + pitchWidth - 45, sPt.x);
                const hitY = sPt.y;

                ctx.save();
                ctx.beginPath();
                ctx.arc(hitX, hitY, 14, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(hitX, hitY, 6, 0, Math.PI * 2);
                ctx.fillStyle = '#ef4444';
                ctx.shadowColor = '#ef4444';
                ctx.shadowBlur = 10;
                ctx.fill();

                // Callout Badge
                ctx.font = '700 10px Inter, sans-serif';
                const hitText = '🎯 WICKET HIT';
                const htW = ctx.measureText(hitText).width;
                const htagX = Math.max(pitchX + 10, Math.min(pitchX + pitchWidth - htW - 20, hitX - htW / 2));
                const htagY = (hitY < midPitchY) ? (hitY + 24) : (hitY - 24);

                ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
                ctx.lineWidth = 1;
                if (ctx.roundRect) {
                    ctx.beginPath();
                    ctx.roundRect(htagX - 6, htagY - 11, htW + 12, 18, 4);
                    ctx.fill();
                    ctx.stroke();
                } else {
                    ctx.fillRect(htagX - 6, htagY - 11, htW + 12, 18);
                }
                ctx.fillStyle = '#f87171';
                ctx.fillText(hitText, htagX, htagY + 2);
                ctx.restore();
            }
        }
    });

    // 2. Draw Bounce Shockwave Events for the active shots
    const bounces = telemetry.bounce_events || [];
    bounces.forEach(b => {
        if (targetShotNum !== null && b.shot && b.shot !== targetShotNum) {
            return;
        }

        let bx, by;
        const matchingShotPts = shotsMap[b.shot || 1];
        if (matchingShotPts && matchingShotPts.length > 1) {
            const mStart = matchingShotPts[0].frame;
            const mSpan = Math.max(1, matchingShotPts[matchingShotPts.length - 1].frame - mStart);
            const bProg = (b.frame - mStart) / mSpan;
            bx = creaseStart + Math.max(0, Math.min(1, bProg)) * playLength;
        } else {
            bx = creaseStart + 0.68 * playLength;
        }
        const latNorm = b.position ? ((b.position[0] - (w * 0.5)) / (w * 0.5)) : 0;
        by = midPitchY + Math.max(-55, Math.min(55, latNorm * 45));

        // Radar shockwave ripple
        ctx.save();
        ctx.beginPath();
        ctx.arc(bx, by, 20, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(bx, by, 12, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(bx, by, 5.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ef4444';
        ctx.shadowColor = '#ef4444';
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.restore();

        // Bounce Callout Badge
        const bText = `💥 Bounce ${b.speed_kmh ? b.speed_kmh + ' km/h' : ''} ${b.angle ? '(' + b.angle + '°)' : ''}`;
        ctx.save();
        ctx.font = '700 10px Inter, sans-serif';
        const tW = ctx.measureText(bText).width;
        const tagX = Math.max(pitchX + 10, Math.min(pitchX + pitchWidth - tW - 20, bx - tW / 2));
        const tagY = (by < midPitchY) ? (by + 26) : (by - 26);

        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
        ctx.lineWidth = 1;
        if (ctx.roundRect) {
            ctx.beginPath();
            ctx.roundRect(tagX - 6, tagY - 11, tW + 12, 18, 4);
            ctx.fill();
            ctx.stroke();
        } else {
            ctx.fillRect(tagX - 6, tagY - 11, tW + 12, 18);
        }
        ctx.fillStyle = '#fef08a';
        ctx.fillText(bText, tagX, tagY + 2);
        ctx.restore();
    });

    // 3. Draw Predicted Future Trajectory Arc
    const predFrame = (telemetry.frame_data || []).slice().reverse().find(f => {
        const matchShot = (targetShotNum === null) || (f.shot === targetShotNum);
        return matchShot && f.predicted_points && f.predicted_points.length > 1;
    });

    if (predFrame && predFrame.predicted_points && predFrame.predicted_points.length > 1) {
        const activeShotPts = shotsMap[targetShotNum || 1];
        if (activeShotPts && activeShotPts.length > 0) {
            const fStart = activeShotPts[0].frame;
            const fSpan = Math.max(1, activeShotPts[activeShotPts.length - 1].frame - fStart);
            const lastProg = (activeShotPts[activeShotPts.length - 1].frame - fStart) / fSpan;
            const lastX = creaseStart + Math.max(0, Math.min(1, lastProg)) * playLength;
            const lastLatNorm = (activeShotPts[activeShotPts.length - 1].x - (w * 0.5)) / (w * 0.5);
            const lastY = midPitchY + Math.max(-55, Math.min(55, lastLatNorm * 45));

            const predPts = predFrame.predicted_points;
            const predMapped = predPts.map((p, idx) => {
                const stepRatio = (idx + 1) / predPts.length;
                const px = Math.min(pitchX + pitchWidth - 35, lastX + stepRatio * (pitchX + pitchWidth - 45 - lastX));
                const pLatNorm = (p[0] - (w * 0.5)) / (w * 0.5);
                const py = midPitchY + Math.max(-55, Math.min(55, pLatNorm * 45));
                return { x: px, y: py };
            });

            if (predMapped.length > 0) {
                ctx.save();
                ctx.strokeStyle = '#34d399';
                ctx.lineWidth = 2.5;
                ctx.setLineDash([5, 4]);
                ctx.shadowColor = 'rgba(52, 211, 153, 0.5)';
                ctx.shadowBlur = 6;
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                predMapped.forEach(pt => ctx.lineTo(pt.x, pt.y));
                ctx.stroke();
                ctx.setLineDash([]);

                const endPt = predMapped[predMapped.length - 1];
                ctx.beginPath();
                ctx.arc(endPt.x, endPt.y, 4.5, 0, Math.PI * 2);
                ctx.fillStyle = '#34d399';
                ctx.fill();
                ctx.restore();
            }
        }
    }

    // 4. Canvas Top-Right HUD Badge
    ctx.save();
    const isBest = (targetShotNum !== null && window.currentInsights?.best_shot === targetShotNum);
    const activeShotInfo = (telemetry.shots_data || []).find(s => s.shot === targetShotNum) || {};
    const activeRelSpeed = activeShotInfo.release_speed_kmh || (telemetry.speed_summary?.release_speed_kmh);

    let badgeText = '';
    if (activeShot === 'all') {
        badgeText = `🌐 ALL ${totalShots} SHOTS OVERLAY`;
    } else if (isBest) {
        badgeText = `⭐ BEST SHOT • SHOT ${targetShotNum} (${activeRelSpeed || '--'} km/h)`;
    } else {
        badgeText = `🏏 SHOT ${targetShotNum} OF ${totalShots} • ${activeRelSpeed || '--'} km/h`;
    }

    ctx.font = '700 11px Inter, sans-serif';
    const bW = ctx.measureText(badgeText).width + 24;
    const bH = 26;
    const bX = canvas.width - bW - 20;
    const bY = 16;

    ctx.fillStyle = isBest ? 'rgba(245, 158, 11, 0.22)' : 'rgba(15, 23, 42, 0.8)';
    ctx.strokeStyle = isBest ? '#f59e0b' : 'rgba(255, 255, 255, 0.18)';
    ctx.lineWidth = 1.2;
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(bX, bY, bW, bH, 13);
        ctx.fill();
        ctx.stroke();
    } else {
        ctx.fillRect(bX, bY, bW, bH);
        ctx.strokeRect(bX, bY, bW, bH);
    }

    ctx.fillStyle = isBest ? '#fbbf24' : '#e2e8f0';
    ctx.fillText(badgeText, bX + 12, bY + 17);
    ctx.restore();
}

function renderTelemetryTable(frameData) {
    const tbody = document.getElementById('telemetryTableBody');
    tbody.innerHTML = '';

    if (!frameData || frameData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center">No telemetry data available</td></tr>';
        return;
    }

    const fps = currentTelemetry.fps || 30.0;
    frameData.forEach(f => {
        const tr = document.createElement('tr');
        const timeSec = (f.frame / fps).toFixed(2) + 's';
        const detectedBadge = f.detected 
            ? '<span style="color: #10b981; font-weight: 600;">● Tracked</span>' 
            : '<span style="color: var(--text-dim);">- None</span>';
        
        const centroidText = f.centroid ? `(${f.centroid[0]}, ${f.centroid[1]})` : '-';
        const speedKmhText = (f.speed_kmh && f.speed_kmh > 0) ? `<strong style="color: var(--primary);">${f.speed_kmh}</strong>` : '-';
        const speedMphText = (f.speed_mph && f.speed_mph > 0) ? `<span style="color: var(--text-muted);">${f.speed_mph}</span>` : '-';
        const bboxText = f.bbox ? `[${f.bbox.join(', ')}]` : '-';
        const eventText = f.is_bounce ? '<span style="color: #ef4444; font-weight: 700;">🏏 BOUNCE</span>' : '-';

        const shotBadge = f.shot ? `<small style="color: var(--primary); font-weight: 700; margin-left: 4px;">S${f.shot}</small>` : '';
        tr.innerHTML = `
            <td>#${f.frame} ${shotBadge}</td>
            <td>${timeSec}</td>
            <td>${detectedBadge}</td>
            <td>${centroidText}</td>
            <td>${speedKmhText}</td>
            <td>${speedMphText}</td>
            <td>${bboxText}</td>
            <td>${f.angle}°</td>
            <td>${eventText}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderBounceLog(bounces) {
    const container = document.getElementById('bounceLogContent');
    container.innerHTML = '';

    if (!bounces || bounces.length === 0) {
        container.innerHTML = '<div class="empty-state">No bounce events detected in this delivery trajectory.</div>';
        return;
    }

    bounces.forEach((b, idx) => {
        const item = document.createElement('div');
        item.className = 'bounce-item';
        item.innerHTML = `
            <div>
                <strong>🏏 Bounce Event #${idx + 1}</strong>: Frame #${b.frame}
            </div>
            <div>
                Deflection: <strong>${b.angle}°</strong> | Speed: <strong>${b.speed_kmh || '--'} km/h</strong>
            </div>
        `;
        container.appendChild(item);
    });
}

function switchAnalyticsTab(tabName) {
    const tabMap = {
        'pitch': { btn: 'tabPitchBtn', pane: 'tabPitch' },
        'telemetry': { btn: 'tabTelemetryBtn', pane: 'tabTelemetry' },
        'bounces': { btn: 'tabBouncesBtn', pane: 'tabBounces' },
        'history': { btn: 'tabHistoryBtn', pane: 'tabHistory' }
    };

    document.querySelectorAll('.pill-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));

    if (tabMap[tabName]) {
        const btnElem = document.getElementById(tabMap[tabName].btn);
        const paneElem = document.getElementById(tabMap[tabName].pane);
        if (btnElem) btnElem.classList.add('active');
        if (paneElem) paneElem.classList.remove('hidden');
    }

    if (tabName === 'pitch') {
        if (currentTelemetry) drawPitchTrajectory(currentTelemetry, window.currentActiveShot || 1);
        else initPitchCanvas();
    } else if (tabName === 'history') {
        loadHistory(false);
    }
}

function exportTelemetryJSON() {
    if (!currentTelemetry) {
        alert('Please run trajectory prediction first!');
        return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentTelemetry, null, 2));
    const a = document.createElement('a');
    a.setAttribute("href", dataStr);
    a.setAttribute("download", "cricket_trajectory_telemetry.json");
    document.body.appendChild(a);
    a.click();
    a.remove();
}

function exportTelemetryCSV() {
    if (!currentTelemetry || !currentTelemetry.frame_data) {
        alert('Please run trajectory prediction first!');
        return;
    }
    let csv = "Frame,Timestamp_sec,Detected,Centroid_X,Centroid_Y,Speed_kmh,Speed_mph,Angle_deg,Is_Bounce\n";
    const fps = currentTelemetry.fps || 30.0;
    currentTelemetry.frame_data.forEach(f => {
        const timeSec = (f.frame / fps).toFixed(2);
        const cx = f.centroid ? f.centroid[0] : "";
        const cy = f.centroid ? f.centroid[1] : "";
        const skmh = f.speed_kmh || 0;
        const smph = f.speed_mph || 0;
        csv += `${f.frame},${timeSec},${f.detected ? 1 : 0},${cx},${cy},${skmh},${smph},${f.angle},${f.is_bounce ? 1 : 0}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "cricket_trajectory_telemetry.csv";
    document.body.appendChild(a);
    a.click();
}

// ─────────────────────────────────────────────
//  AI INSIGHTS
// ─────────────────────────────────────────────
async function runAIAnalysis() {
    if (!lastCompletedTaskId) return;

    const btn = document.getElementById('analyzeBtn');
    const body = document.getElementById('aiInsightsBody');

    btn.disabled = true;
    btn.textContent = '⏳ Analysing...';
    body.innerHTML = `
        <div class="ai-loading">
            <div class="spinner"></div>
            Sending delivery telemetry to Gemini AI — generating shot-wise coaching insights...
        </div>`;

    try {
        const res = await fetch(`/api/analyze/${lastCompletedTaskId}`, { method: 'POST' });
        let data;
        const rawText = await res.text();
        try {
            data = JSON.parse(rawText);
        } catch (parseErr) {
            body.innerHTML = `<div class="ai-empty-state" style="color:#ef4444">⚠️ Server returned an error (Status ${res.status}). Please try again.</div>`;
            return;
        }

        if (!res.ok || data.error) {
            body.innerHTML = `<div class="ai-empty-state" style="color:#ef4444">⚠️ ${data.error || 'Analysis failed'}</div>`;
            return;
        }

        window.currentInsights = data.insights;
        if (data.shots_data && data.shots_data.length > 0) {
            window.currentShotsData = data.shots_data;
            if (currentTelemetry) {
                currentTelemetry.shots_data = data.shots_data;
            }
        }

        // Re-render dashboard grid so insights and best shot badges update immediately
        if (window.currentShotsData && window.currentShotsData.length > 1) {
            renderShotsDashboard(window.currentTaskData, window.currentShotsData);
        }

        window.currentActiveShot = (window.currentShotsData && window.currentShotsData.length > 1) ? (data.insights.best_shot || 'best') : 1;
        renderShotSelector(window.currentShotsData, window.currentInsights, window.currentActiveShot);
        renderActiveShotContent(window.currentActiveShot);
        if (currentTelemetry) {
            drawPitchTrajectory(currentTelemetry, window.currentActiveShot);
        }

        // Update history catalog so AI badge shows
        loadHistory();

    } catch (e) {
        body.innerHTML = `<div class="ai-empty-state" style="color:#ef4444">⚠️ Request failed: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Re-analyze Insights';
    }
}

/** Clean up any LaTeX notation and stray dollar signs */
function cleanLatex(text) {
    if (!text) return '';
    return text
        // $50.9^\circ$ or $50^\circ$ or 50^\circ -> 50.9°
        .replace(/\$?([0-9.]+)\s*\\?\^?\\circ\$?/g, '$1°')
        // $151.2\text{ km/h}$ -> 151.2 km/h
        .replace(/\$([0-9.]+)\s*\\text\{\s*(.+?)\s*\}\$/g, '$1 $2')
        .replace(/\\text\{\s*(.+?)\s*\}/g, '$1')
        // $0.4$ -> 0.4
        .replace(/\$([0-9.]+)\$/g, '$1')
        // Any remaining $formula$ -> formula
        .replace(/\$([^$\n]+)\$/g, '$1');
}

/** Very lightweight markdown → HTML renderer (no external libs needed) */
function simpleMarkdown(md) {
    if (!md) return '';
    md = cleanLatex(md);
    return md
        // headings
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // unordered list items
        .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
        // wrap consecutive <li> in <ul>
        .replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
        // numbered list
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // line breaks → paragraphs
        .split(/\n{2,}/)
        .map(block => {
            block = block.trim();
            if (!block) return '';
            if (block.startsWith('<h') || block.startsWith('<ul') || block.startsWith('<li')) return block;
            return `<p>${block.replace(/\n/g, '<br>')}</p>`;
        })
        .join('');
}

// ─────────────────────────────────────────────
//  PROCESSED VIDEO HISTORY MANAGEMENT
// ─────────────────────────────────────────────

function formatTimestamp(isoStr) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return isoStr;
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        return isoStr;
    }
}

function formatTimeAgo(isoStr) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        const diffMs = Date.now() - d.getTime();
        const sec = Math.floor(diffMs / 1000);
        if (sec < 60) return 'Just now';
        const min = Math.floor(sec / 60);
        if (min < 60) return `${min}m ago`;
        const hr = Math.floor(min / 60);
        if (hr < 24) return `${hr}h ago`;
        const days = Math.floor(hr / 24);
        return `${days}d ago`;
    } catch (e) {
        return '';
    }
}

async function loadHistory(autoSelect = false) {
    try {
        const res = await fetch('/api/history');
        if (!res.ok) return;
        const data = await res.json();
        allHistoryRecords = data.history || [];

        const countBadge = document.getElementById('historyCountBadge');
        if (countBadge) countBadge.innerText = allHistoryRecords.length;

        const sidebarCount = document.getElementById('sidebarHistoryCount');
        if (sidebarCount) sidebarCount.innerText = allHistoryRecords.length;

        const topNavBadge = document.getElementById('topNavHistoryCount');
        if (topNavBadge) topNavBadge.innerText = allHistoryRecords.length;

        const searchInput = document.getElementById('historySearchInput');
        const q = searchInput ? searchInput.value : '';
        if (q) {
            filterHistory(q);
        } else {
            renderHistory(allHistoryRecords);
        }
        renderSidebarHistory(allHistoryRecords);

        if (autoSelect && allHistoryRecords.length > 0) {
            switchAnalyticsTab('history');
        }
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function renderHistory(items) {
    const grid = document.getElementById('historyGrid');
    if (!grid) return;

    if (!items || items.length === 0) {
        grid.innerHTML = `
            <div class="history-empty">
                <div class="history-empty-icon">📁</div>
                <div class="history-empty-text">No delivery history records found.</div>
            </div>`;
        return;
    }

    grid.innerHTML = items.map(item => {
        const isActive = item.task_id === currentLoadedHistoryTaskId;
        const relSpeed = item.release_speed_kmh ? `${item.release_speed_kmh} km/h` : (item.avg_speed_kmh ? `${item.avg_speed_kmh} km/h` : '--');
        const avgPeak = (item.avg_speed_kmh || item.max_speed_kmh) ? `${item.avg_speed_kmh || '--'} / ${item.max_speed_kmh || '--'}` : '--';
        const bounceCount = item.bounce_count !== undefined ? item.bounce_count : 0;
        const framesText = item.total_frames ? `${item.detected_frames || 0}/${item.total_frames}` : '--';
        const displayName = item.input_filename && item.input_filename !== 'unknown' ? item.input_filename : item.output_filename;

        return `
            <div class="history-card ${isActive ? 'active' : ''}" id="history-card-${item.task_id}">
                <div class="history-card-header">
                    <div class="history-card-title-group">
                        <span class="history-file-icon">📹</span>
                        <div class="history-file-meta">
                            <div class="history-file-name" title="${displayName}">${displayName}</div>
                            <div class="history-timestamp">${formatTimestamp(item.timestamp)}</div>
                        </div>
                    </div>
                    <div class="history-badges">
                        ${item.has_insights ? '<span class="history-badge history-badge-ai" title="AI Coaching Insights Available">🤖 AI Insights</span>' : ''}
                        <span class="history-badge history-badge-id">#${item.task_id}</span>
                    </div>
                </div>
                <div class="history-card-metrics">
                    <div class="history-metric">
                        <span class="history-metric-lbl">Release</span>
                        <span class="history-metric-val highlight">${relSpeed}</span>
                    </div>
                    <div class="history-metric">
                        <span class="history-metric-lbl">Avg / Peak</span>
                        <span class="history-metric-val">${avgPeak}</span>
                    </div>
                    <div class="history-metric">
                        <span class="history-metric-lbl">Bounces</span>
                        <span class="history-metric-val">${bounceCount}</span>
                    </div>
                    <div class="history-metric">
                        <span class="history-metric-lbl">Tracked</span>
                        <span class="history-metric-val">${framesText}</span>
                    </div>
                </div>
                <div class="history-card-actions">
                    <button class="btn-small btn-primary history-inspect-btn" onclick="loadHistoryItem('${item.task_id}')">
                        ⚡ Inspect Run
                    </button>
                    <a class="btn-small btn-secondary" href="/videos/processed/${item.output_filename}" download title="Download Processed Video">
                        📥 Video
                    </a>
                    <button class="btn-small btn-danger-icon" onclick="deleteHistoryItem('${item.task_id}', event)" title="Delete record">
                        🗑️
                    </button>
                </div>
            </div>`;
    }).join('');
}

function renderSidebarHistory(items) {
    const list = document.getElementById('sidebarHistoryList');
    if (!list) return;

    if (!items || items.length === 0) {
        list.innerHTML = '<div class="recent-empty">No deliveries processed yet</div>';
        return;
    }

    const recent = items.slice(0, 6);
    list.innerHTML = recent.map(item => {
        const isActive = item.task_id === currentLoadedHistoryTaskId;
        const displayName = item.input_filename && item.input_filename !== 'unknown' ? item.input_filename : item.output_filename;
        const speed = item.release_speed_kmh ? `${item.release_speed_kmh}k` : (item.avg_speed_kmh ? `${item.avg_speed_kmh}k` : '');

        return `
            <div class="recent-item ${isActive ? 'active' : ''}" id="recent-item-${item.task_id}" onclick="loadHistoryItem('${item.task_id}')" title="Inspect ${displayName}">
                <div class="recent-item-info">
                    <div class="recent-item-name">${displayName}</div>
                    <div class="recent-item-time">${formatTimeAgo(item.timestamp)}</div>
                </div>
                <div class="recent-item-stats">
                    ${speed ? `<span class="recent-speed-chip">${speed}</span>` : ''}
                    ${item.has_insights ? '<span class="recent-ai-dot" title="Gemini AI Insights Available">🤖</span>' : ''}
                </div>
            </div>`;
    }).join('');
}

async function loadHistoryItem(taskId) {
    currentLoadedHistoryTaskId = taskId;

    // Visual indicators
    document.querySelectorAll('.history-card').forEach(c => c.classList.remove('active'));
    const activeCard = document.getElementById(`history-card-${taskId}`);
    if (activeCard) activeCard.classList.add('active');

    document.querySelectorAll('.recent-item').forEach(r => r.classList.remove('active'));
    const activeRecent = document.getElementById(`recent-item-${taskId}`);
    if (activeRecent) activeRecent.classList.add('active');

    try {
        const res = await fetch(`/api/history/${taskId}`);
        if (!res.ok) {
            alert('Failed to load history record: ' + res.statusText);
            return;
        }
        const data = await res.json();
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        displayResults(data);

        // Switch to Pitch Coordinate Map tab to inspect loaded trajectory
        switchAnalyticsTab('pitch');

        // Scroll smoothly to video player
        const videoWrapper = document.getElementById('videoWrapper');
        if (videoWrapper) {
            videoWrapper.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    } catch (e) {
        console.error('Failed to load historical delivery:', e);
        alert('Error loading historical delivery: ' + e.message);
    }
}

async function deleteHistoryItem(taskId, event) {
    if (event) event.stopPropagation();
    if (!confirm('Are you sure you want to delete this delivery run from history?')) return;

    try {
        const res = await fetch(`/api/history/${taskId}?delete_files=true`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            await loadHistory();
        } else {
            alert('Failed to delete: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        alert('Error deleting history item: ' + e.message);
    }
}

async function confirmClearHistory() {
    if (!confirm('Clear all processed delivery history records? (Video files will remain on disk)')) return;
    try {
        const res = await fetch('/api/history/clear', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            await loadHistory();
        }
    } catch (e) {
        alert('Error clearing history: ' + e.message);
    }
}

function filterHistory(query) {
    const q = (query || '').toLowerCase().trim();
    const clearBtn = document.getElementById('historyClearSearch');
    if (clearBtn) {
        if (q) clearBtn.classList.remove('hidden');
        else clearBtn.classList.add('hidden');
    }

    if (!q) {
        renderHistory(allHistoryRecords);
        return;
    }

    const filtered = allHistoryRecords.filter(item => {
        const nameMatch = (item.input_filename || '').toLowerCase().includes(q) ||
                          (item.output_filename || '').toLowerCase().includes(q);
        const idMatch = (item.task_id || '').toLowerCase().includes(q);
        const dateMatch = (item.timestamp || '').toLowerCase().includes(q);
        const speedMatch = `${item.release_speed_kmh || ''} ${item.avg_speed_kmh || ''} ${item.max_speed_kmh || ''}`.includes(q);
        return nameMatch || idMatch || dateMatch || speedMatch;
    });

    renderHistory(filtered);
}

function clearHistorySearch() {
    const input = document.getElementById('historySearchInput');
    if (input) input.value = '';
    filterHistory('');
}
