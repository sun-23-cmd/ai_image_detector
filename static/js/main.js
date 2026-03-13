// ── AI Image Detector · main.js ──────────────────────────────────────────────

const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const btnAnalyze   = document.getElementById('btnAnalyze');
const btnRetrain   = document.getElementById('btnRetrain');
const progressWrap = document.getElementById('progressWrap');
const progressBar  = document.getElementById('progressBar');
const errorBox     = document.getElementById('errorBox');
const results      = document.getElementById('results');
const fbCorrect    = document.getElementById('fbCorrect');
const fbWrong      = document.getElementById('fbWrong');

let selectedFile = null;
let lastFeatures = null;
let lastLabel    = null;

// Load stats on page load
loadStats();

async function loadStats() {
  try {
    const r = await fetch('/stats');
    const d = await r.json();
    updateStats(d.training_count, d.model_active);
  } catch(e) {}
}

function updateStats(count, modelActive) {
  document.getElementById('statSamples').textContent = count;
  const dot   = document.getElementById('modelDot');
  const label = document.getElementById('statModel');
  if (modelActive) {
    label.textContent = 'ML (trained)';
    dot.classList.add('dot-active');
  } else {
    label.textContent = 'Heuristic';
    dot.classList.remove('dot-active');
  }
}

// ── Drag and drop ─────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('border-cyan-400', 'bg-cyan-400/5');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('border-cyan-400', 'bg-cyan-400/5');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('border-cyan-400', 'bg-cyan-400/5');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  document.getElementById('uploadLabel').textContent = '✅ ' + file.name;
  document.getElementById('uploadHint').textContent  = (file.size / 1024 / 1024).toFixed(2) + ' MB';
  btnAnalyze.disabled = false;
  btnAnalyze.classList.remove('opacity-40', 'cursor-not-allowed');
  hideError();
  results.style.display = 'none';
}

// ── Analyze ───────────────────────────────────────────────────────────────────
btnAnalyze.addEventListener('click', analyze);

async function analyze() {
  if (!selectedFile) return;

  btnAnalyze.disabled = true;
  btnAnalyze.classList.add('opacity-40', 'cursor-not-allowed');
  showProgress();
  hideError();
  results.style.display = 'none';
  dropZone.classList.add('scanning');

  const form = new FormData();
  form.append('image', selectedFile);

  try {
    animateProgress(0, 65, 600);
    const resp = await fetch('/analyze', { method: 'POST', body: form });
    const data = await resp.json();
    animateProgress(65, 100, 300);

    setTimeout(() => {
      hideProgress();
      dropZone.classList.remove('scanning');
      if (data.error) showError(data.error);
      else            showResults(data);
      btnAnalyze.disabled = false;
      btnAnalyze.classList.remove('opacity-40', 'cursor-not-allowed');
    }, 350);
  } catch(err) {
    hideProgress();
    dropZone.classList.remove('scanning');
    showError('Could not reach server. Make sure Flask is running on port 5000.');
    btnAnalyze.disabled = false;
    btnAnalyze.classList.remove('opacity-40', 'cursor-not-allowed');
  }
}

// ── Show Results ──────────────────────────────────────────────────────────────
function showResults(data) {
  const isAI   = data.label === 'AI Generated';
  lastFeatures = data.features;
  lastLabel    = data.label;

  // Verdict banner
  const banner = document.getElementById('verdictBanner');
  banner.className = 'flex items-center gap-4 flex-wrap p-6 border-b border-slate-700 ' +
                     (isAI ? 'verdict-ai' : 'verdict-real');

  document.getElementById('verdictIcon').textContent  = isAI ? '🤖' : '📷';
  document.getElementById('verdictLabel').textContent = data.label;
  document.getElementById('verdictLabel').className   = 'text-2xl font-bold verdict-label';
  document.getElementById('verdictSub').textContent   = 'AI probability: ' + data.ai_probability + '%';

  // Method badge
  const badge = document.getElementById('methodBadge');
  badge.textContent = data.method === 'ml' ? '🧠 ML Model' : '📐 Heuristic';
  badge.className   = 'ml-auto px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest ' +
                      (data.method === 'ml' ? 'badge-ml' : 'badge-heuristic');

  // Confidence bar
  const confBar = document.getElementById('confBar');
  confBar.className = 'conf-bar-fill ' + (isAI ? 'conf-bar-ai' : 'conf-bar-real');
  document.getElementById('confPct').textContent = data.confidence + '%';
  setTimeout(() => { confBar.style.width = data.confidence + '%'; }, 50);

  // Preview image
  document.getElementById('previewImg').src = data.preview;

  // Signals list
  const list = document.getElementById('signalsList');
  list.innerHTML = '';
  (data.signals || []).forEach(function(s) {
    const type = s[0], msg = s[1];
    const pill = document.createElement('div');
    pill.className = 'flex items-start gap-3 bg-slate-900 rounded-lg p-3 text-xs leading-relaxed border ' +
                     (type === 'AI' ? 'signal-ai' : 'signal-real');
    pill.innerHTML = `<span class="w-2 h-2 rounded-full mt-1 shrink-0 ${type === 'AI' ? 'dot-ai-color' : 'dot-real-color'}"></span>
                      <span>${msg}</span>`;
    list.appendChild(pill);
  });

  // Feedback reset
  fbCorrect.disabled = false;
  fbWrong.disabled   = false;
  fbCorrect.classList.remove('opacity-30', 'cursor-not-allowed', 'scale-95');
  fbWrong.classList.remove('opacity-30', 'cursor-not-allowed', 'scale-95');
  document.getElementById('feedbackSaved').style.display = 'none';
  document.getElementById('feedbackSaved').className = 'feedback-saved text-sm text-emerald-400';

  updateStats(data.training_count, data.model_active);

  results.style.display = 'block';
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Feedback ──────────────────────────────────────────────────────────────────
fbCorrect.addEventListener('click', () => sendFeedback(lastLabel));
fbWrong.addEventListener('click', () => {
  const flipped = lastLabel === 'AI Generated' ? 'Real / Authentic' : 'AI Generated';
  sendFeedback(flipped);
});

async function sendFeedback(correctLabel) {
  if (!lastFeatures) return;

  // Disable and dim both buttons immediately
  fbCorrect.disabled = true;
  fbWrong.disabled   = true;
  fbCorrect.classList.add('opacity-30', 'cursor-not-allowed', 'scale-95');
  fbWrong.classList.add('opacity-30', 'cursor-not-allowed', 'scale-95');

  try {
    const resp = await fetch('/feedback', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ features: lastFeatures, correct_label: correctLabel })
    });
    const data = await resp.json();
    if (data.saved) {
      const saved = document.getElementById('feedbackSaved');
      saved.style.display = 'inline';
      updateStats(data.total, document.getElementById('statModel').textContent !== 'Heuristic');
      showToast('✓ Saved! Total samples: ' + data.total, 'success');
    }
  } catch(e) {
    showToast('Failed to save feedback.', 'error');
    // Re-enable on error so user can try again
    fbCorrect.disabled = false;
    fbWrong.disabled   = false;
    fbCorrect.classList.remove('opacity-30', 'cursor-not-allowed', 'scale-95');
    fbWrong.classList.remove('opacity-30', 'cursor-not-allowed', 'scale-95');
  }
}

// ── Retrain ───────────────────────────────────────────────────────────────────
btnRetrain.addEventListener('click', async () => {
  btnRetrain.disabled = true;
  btnRetrain.textContent = '⏳ Training...';

  try {
    const resp = await fetch('/retrain', { method: 'POST' });
    const data = await resp.json();
    if (data.success) {
      showToast('🎉 ' + data.message, 'success');
      updateStats(data.training_count, true);
    } else {
      showToast('⚠️ ' + data.message, 'error');
    }
  } catch(e) {
    showToast('Failed to retrain.', 'error');
  }

  btnRetrain.disabled = false;
  btnRetrain.textContent = '🧠 Retrain Model on Saved Data';
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showProgress() { progressWrap.style.display = 'block'; progressBar.style.width = '0%'; }
function hideProgress() { progressWrap.style.display = 'none'; }

function animateProgress(from, to, ms) {
  const step = (to - from) / (ms / 30);
  let cur = from;
  const id = setInterval(() => {
    cur += step;
    if (cur >= to) { cur = to; clearInterval(id); }
    progressBar.style.width = cur + '%';
  }, 30);
}

function showError(msg) {
  errorBox.textContent   = '⚠️ ' + msg;
  errorBox.style.display = 'block';
}
function hideError() { errorBox.style.display = 'none'; }

let toastTimer;
function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = 'toast px-6 py-3 rounded-full text-sm font-mono border bg-slate-800 ' +
                  (type === 'success' ? 'border-emerald-400 text-emerald-400' : 'border-red-400 text-red-400');
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 3000);
}