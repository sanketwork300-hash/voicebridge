import { DEFAULT_SETTINGS } from '../common/protocol.js';

const $ = (id) => document.getElementById(id);

const PRESET_DEFAULTS = {
  anime_drama: {
    sourceLanguage: 'auto', targetLanguage: 'en', mode: 'subtitles', profile: 'balanced',
    dualSubtitles: true, nameRendering: 'preserve_and_romanize', honorifics: 'preserve_honorifics',
  },
  kpop_live: {
    sourceLanguage: 'ko', targetLanguage: 'en', mode: 'subtitles', profile: 'low_latency',
    dualSubtitles: true, nameRendering: 'translate', honorifics: 'preserve_honorifics',
  },
  conference: {
    sourceLanguage: 'en', targetLanguage: 'hi', mode: 'speech_and_subtitles',
    profile: 'balanced', dualSubtitles: false,
  },
};

function radio(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}
function setRadio(name, value) {
  const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (el) el.checked = true;
}

function readForm() {
  return {
    preset: $('preset').value,
    sourceLanguage: $('source').value,
    targetLanguage: $('target').value,
    mode: radio('mode'),
    profile: radio('profile'),
    dualSubtitles: $('dual').checked,
    originalAudio: $('originalAudio').value,
    nameRendering: $('names').value,
    honorifics: $('honorifics').value,
    subtitleSize: Number($('size').value),
    serverUrl: $('server').value.trim() || DEFAULT_SETTINGS.serverUrl,
    apiToken: $('token').value,
  };
}

function writeForm(s) {
  $('preset').value = s.preset || '';
  $('source').value = s.sourceLanguage;
  $('target').value = s.targetLanguage;
  setRadio('mode', s.mode);
  setRadio('profile', s.profile);
  $('dual').checked = !!s.dualSubtitles;
  $('originalAudio').value = s.originalAudio;
  $('names').value = s.nameRendering;
  $('honorifics').value = s.honorifics;
  $('size').value = s.subtitleSize;
  $('sizeOut').value = s.subtitleSize;
  $('server').value = s.serverUrl;
  $('token').value = s.apiToken || '';
}

async function persist() {
  await chrome.runtime.sendMessage({ type: 'settings:set', settings: readForm() });
}

function showError(message) {
  const el = $('error');
  if (!message) { el.hidden = true; el.textContent = ''; return; }
  el.hidden = false;
  el.textContent = message;
}

function dot(id, cls) { $(id).className = `dot${cls ? ' ' + cls : ''}`; }

function applyState(state) {
  const capturing = !!state?.capturing;
  $('start').hidden = capturing;
  $('pause').hidden = !capturing;
  $('stop').hidden = !capturing;
  $('pause').textContent = state?.paused ? 'Resume' : 'Pause';
  $('badge').textContent = capturing ? (state.paused ? 'paused' : 'live') : 'idle';
  $('badge').classList.toggle('on', capturing && !state.paused);

  const ws = state?.wsState || 'idle';
  $('connText').textContent = ws;
  dot('dConn', ws === 'connected' ? 'on' : (ws === 'error' || ws === 'closed') ? 'err'
      : ws === 'reconnecting' ? 'warn' : '');
  $('capText').textContent = capturing ? (state.paused ? 'paused' : 'capturing') : 'idle';
  dot('dCap', capturing && !state.paused ? 'on' : '');
  $('langText').textContent = state?.language || '—';
  dot('dLang', state?.language ? 'on' : '');
  showError(state?.lastError || '');
}

async function refresh() {
  const res = await chrome.runtime.sendMessage({ type: 'capture:status' });
  if (res?.settings) writeForm({ ...DEFAULT_SETTINGS, ...res.settings });
  applyState(res?.state);
}

$('preset').addEventListener('change', async () => {
  const preset = $('preset').value;
  if (preset && PRESET_DEFAULTS[preset]) {
    writeForm({ ...DEFAULT_SETTINGS, ...readForm(), ...PRESET_DEFAULTS[preset], preset });
  }
  await persist();
});

for (const id of ['source', 'target', 'dual', 'originalAudio', 'names', 'honorifics', 'server', 'token']) {
  $(id).addEventListener('change', persist);
}
for (const name of ['mode', 'profile']) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((el) =>
    el.addEventListener('change', persist));
}
$('size').addEventListener('input', () => { $('sizeOut').value = $('size').value; });
$('size').addEventListener('change', persist);

$('start').addEventListener('click', async () => {
  showError('');
  $('start').disabled = true;
  await persist();
  const res = await chrome.runtime.sendMessage({ type: 'capture:start' });
  $('start').disabled = false;
  if (!res?.ok) {
    showError(friendlyError(res?.error));
    return;
  }
  applyState(res.state);
});

$('stop').addEventListener('click', async () => {
  const res = await chrome.runtime.sendMessage({ type: 'capture:stop' });
  applyState(res?.state);
});

$('pause').addEventListener('click', async () => {
  const paused = $('pause').textContent === 'Pause';
  const res = await chrome.runtime.sendMessage({
    type: paused ? 'capture:pause' : 'capture:resume',
  });
  applyState(res?.state);
});

function friendlyError(error) {
  const text = String(error || 'unknown error');
  if (/activeTab|permission|not been invoked/i.test(text)) {
    return 'Chrome would not grant tab capture. Click the VoiceBridge icon directly on the tab you want to translate, then press Start.';
  }
  if (/Cannot capture|chrome:\/\/|Web Store/i.test(text)) {
    return 'This page cannot be captured by extensions. Try a normal http(s) page.';
  }
  if (/Failed to fetch|NetworkError|ECONNREFUSED/i.test(text)) {
    return 'Cannot reach the VoiceBridge gateway. Start it with `voicebridge-gateway` and check the Gateway URL under Options.';
  }
  return text;
}

// Keep the popup live while it is open; it is closed most of the time, so a
// short interval is cheap and avoids a stale UI after a reconnect.
refresh();
setInterval(refresh, 1000);
