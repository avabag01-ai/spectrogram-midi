const API_BASE = 'http://localhost:5005';
let selectedFilePath = null;
let midiDataB64 = null;

// UI Elements
const dropZone = document.getElementById('drop-zone');
const fileInfo = document.getElementById('file-info');
const analyzeBtn = document.getElementById('analyze-btn');
const downloadBtn = document.getElementById('download-btn');
const loading = document.getElementById('loading');
const engineStatus = document.getElementById('engine-status');

// Sliders & Toggles
const turboToggle = document.getElementById('turbo-toggle');
const bandToggle = document.getElementById('band-toggle');
const confSlider = document.getElementById('conf-slider');
const rakeSlider = document.getElementById('rake-slider');
const sustainSlider = document.getElementById('sustain-slider');
const durSlider = document.getElementById('dur-slider');
const patchSelect = document.getElementById('patch-select');

// Slider Displays
confSlider.oninput = () => document.getElementById('conf-val').innerText = confSlider.value;
rakeSlider.oninput = () => document.getElementById('rake-val').innerText = rakeSlider.value;
sustainSlider.oninput = () => document.getElementById('sustain-val').innerText = sustainSlider.value;
durSlider.oninput = () => document.getElementById('dur-val').innerText = durSlider.value;

// Toggle behaviors
turboToggle.onclick = () => turboToggle.classList.toggle('active');
bandToggle.onclick = () => bandToggle.classList.toggle('active');

// File Upload Handling
dropZone.onclick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.wav,.mp3';
    input.onchange = (e) => handleFile(e.target.files[0]);
    input.click();
};

async function handleFile(file) {
    if (!file) return;
    fileInfo.innerText = `Target Locked: ${file.name}`;
    engineStatus.innerText = 'Engine: File Loaded';

    loading.style.display = 'flex';
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        selectedFilePath = data.file_path;
        document.getElementById('playback-time').innerText = `00:00 / ${formatTime(data.duration)}`;
    } catch (err) {
        console.error(err);
        alert('Engine connection failed. Make sure internal server is running.');
    } finally {
        loading.style.display = 'none';
    }
}

function formatTime(s) {
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
}

// Analyze Execution
analyzeBtn.onclick = async () => {
    if (!selectedFilePath) {
        alert('Please select an audio file first.');
        return;
    }

    loading.style.display = 'flex';
    engineStatus.innerText = 'Engine: AI Perception Phase...';

    const params = {
        file_path: selectedFilePath,
        turbo_mode: turboToggle.classList.contains('active'),
        full_band_mode: bandToggle.classList.contains('active'),
        rake_sens: parseFloat(rakeSlider.value)
    };

    try {
        await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        await updateMidi();
        engineStatus.innerText = 'Engine: Perception Complete';
    } catch (err) {
        console.error(err);
        alert('Analysis failed.');
    } finally {
        loading.style.display = 'none';
    }
};

async function updateMidi() {
    const params = {
        confidence: parseFloat(confSlider.value),
        sustain: parseInt(sustainSlider.value),
        min_duration: parseInt(durSlider.value),
        patch: parseInt(patchSelect.value)
    };

    try {
        const res = await fetch(`${API_BASE}/filter`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const data = await res.json();
        midiDataB64 = data.midi_b64;
        downloadBtn.disabled = false;
        renderEvents(data.events);
    } catch (err) {
        console.error(err);
    }
}

// Real-time update on slider change (only if analysis exists)
[confSlider, sustainSlider, durSlider, patchSelect].forEach(el => {
    el.onchange = () => {
        if (midiDataB64) updateMidi();
    };
});

function renderEvents(events) {
    const canvas = document.getElementById('viz-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!events || events.length === 0) return;

    const minNote = 40;
    const maxNote = 80;
    const padding = 20;

    events.forEach(evt => {
        const x = (evt.start / 100) * canvas.width; // rough scaling
        const w = ((evt.end - evt.start) / 100) * canvas.width;
        const y = canvas.height - ((evt.note - minNote) / (maxNote - minNote)) * canvas.height;

        ctx.fillStyle = evt.track === 'main' ? '#00ffcc' : '#444';
        ctx.shadowBlur = evt.track === 'main' ? 10 : 0;
        ctx.shadowColor = '#00ffcc';
        ctx.fillRect(x, y, Math.max(w, 5), 8);
    });
}

downloadBtn.onclick = () => {
    if (!midiDataB64) return;

    const binary = atob(midiDataB64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    const blob = new Blob([bytes], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'aegis_output.mid';
    a.click();
};
