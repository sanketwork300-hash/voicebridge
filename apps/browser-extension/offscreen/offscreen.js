// VoiceBridge offscreen document.
//
// Responsibilities:
//   * hold the tab-audio MediaStream obtained from a streamId
//   * keep the tab audible (tabCapture mutes the tab by default)
//   * convert to 16 kHz mono PCM and stream it to the gateway
//   * play translated speech back in sequence
//   * reconnect with exponential backoff
//
// Verified capture pattern (chrome.tabCapture.getMediaStreamId in the service
// worker, getUserMedia with chromeMediaSource:'tab' in an offscreen document
// created with reason USER_MEDIA) follows the approach used by the Kami Subs
// and WhisperLive Chrome extensions. It is the supported MV3 route; a service
// worker cannot call getUserMedia itself.

import { CLIENT, SERVER, SAMPLE_RATE, httpToWs, buildSessionPayload } from '../common/protocol.js';

const MAX_BACKOFF_MS = 15000;
const PING_INTERVAL_MS = 20000;

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let captureNode = null;
let passthroughGain = null;
let socket = null;
let settings = null;
let sessionId = null;

let running = false;          // true between start() and stop()
let reconnectAttempts = 0;
let reconnectTimer = null;
let pingTimer = null;
let playheadTime = 0;         // AudioContext time at which the next dub segment starts
let pendingFrames = [];       // frames captured while the socket is down

// Bounded: if the socket stays down we must not grow this forever. 100 frames
// is 10 s of audio; older audio is worthless for live translation anyway.
const MAX_PENDING_FRAMES = 100;

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
    return true;
  }
  return false;
}

function report(type, payload = {}) {
  chrome.runtime.sendMessage({ target: 'background', type, ...payload }).catch(() => {});
}

// ---------------------------------------------------------------- session

async function createSession() {
  const base = settings.serverUrl.replace(/\/+$/, '');
  const headers = { 'content-type': 'application/json' };
  if (settings.apiToken) headers.authorization = `Bearer ${settings.apiToken}`;

  const res = await fetch(`${base}/v1/sessions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(buildSessionPayload(settings)),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`gateway returned ${res.status}: ${detail.slice(0, 200)}`);
  }
  const data = await res.json();
  return data.session_id;
}

function openSocket() {
  const base = httpToWs(settings.serverUrl.replace(/\/+$/, ''));
  let url = `${base}/v1/sessions/${sessionId}/stream`;
  if (settings.apiToken) url += `?token=${encodeURIComponent(settings.apiToken)}`;

  socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';

  socket.addEventListener('open', () => {
    reconnectAttempts = 0;
    report('ws:state', { state: 'connected' });
    send({ type: CLIENT.SESSION_CONFIG, input: { sample_rate: SAMPLE_RATE, channels: 1 } });
    send({ type: CLIENT.AUDIO_START });
    // Flush whatever was captured while disconnected, oldest first.
    for (const frame of pendingFrames) {
      if (socket.readyState === WebSocket.OPEN) socket.send(frame);
    }
    pendingFrames = [];
    clearInterval(pingTimer);
    pingTimer = setInterval(() => send({ type: CLIENT.PING, echo: Date.now() }), PING_INTERVAL_MS);
  });

  socket.addEventListener('message', (event) => {
    let payload;
    try { payload = JSON.parse(event.data); } catch { return; }
    handleServerEvent(payload);
  });

  socket.addEventListener('error', () => {
    report('ws:state', { state: 'error' });
    if (reconnectAttempts === 0) {
      report('backend:error', {
        message: `Cannot reach VoiceBridge at ${settings.serverUrl}. Is the gateway running?`,
      });
    }
  });

  socket.addEventListener('close', () => {
    clearInterval(pingTimer);
    report('ws:state', { state: 'closed' });
    scheduleReconnect();
  });
}

function scheduleReconnect() {
  if (!running || reconnectTimer) return;
  reconnectAttempts += 1;
  const delay = Math.min(MAX_BACKOFF_MS, 250 * 2 ** (reconnectAttempts - 1));
  report('ws:state', { state: 'reconnecting', attempt: reconnectAttempts, delay });
  reconnectTimer = setTimeout(async () => {
    reconnectTimer = null;
    if (!running) return;
    try {
      // The old session is gone with its socket; make a fresh one so the
      // pipeline starts from a clean transcript rather than a stale buffer.
      sessionId = await createSession();
      openSocket();
    } catch (err) {
      report('backend:error', { message: String(err.message || err) });
      scheduleReconnect();
    }
  }, delay);
}

// ------------------------------------------------------------ server events

function handleServerEvent(ev) {
  switch (ev.event_type) {
    case SERVER.ASR_PARTIAL:
      report('subtitle', { kind: 'partial', source: ev.text, committed: false });
      break;
    case SERVER.ASR_STABLE:
      report('subtitle', { kind: 'stable', source: ev.text, start: ev.start, end: ev.end });
      break;
    case SERVER.TRANSLATION_FINAL:
      report('subtitle', {
        kind: 'translation',
        source: ev.source_text,
        translation: ev.translated_text,
        sourceLanguage: ev.source_language,
        targetLanguage: ev.target_language,
        speaker: ev.speaker,
        committed: true,
      });
      break;
    case SERVER.TTS_AUDIO:
      playTranslatedAudio(ev);
      break;
    case SERVER.LANGUAGE_DETECTED:
      report('language', { language: ev.language, stable: !!ev.stable });
      break;
    case SERVER.WARNING:
      report('backend:warning', { code: ev.code, message: ev.message });
      break;
    case SERVER.ERROR:
      report('backend:error', { message: ev.message, recoverable: ev.recoverable });
      break;
    case SERVER.SESSION_ENDED:
      report('session:ended', { metrics: ev.metrics || {} });
      break;
    default:
      break;
  }
}

// --------------------------------------------------------------- playback

function playTranslatedAudio(ev) {
  if (!audioContext) return;
  const binary = atob(ev.audio);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const pcm = new Int16Array(bytes.buffer);

  const rate = ev.sample_rate || SAMPLE_RATE;
  const buffer = audioContext.createBuffer(1, pcm.length, rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

  const node = audioContext.createBufferSource();
  node.buffer = buffer;
  node.connect(audioContext.destination);

  // The server already guarantees sequence order; scheduling each segment after
  // the previous one prevents overlap when several arrive in one burst.
  playheadTime = Math.max(playheadTime, audioContext.currentTime);
  node.start(playheadTime);
  playheadTime += buffer.duration;

  // Duck the original programme audio while translated speech plays.
  if (settings.originalAudio === 'mixed' && passthroughGain) {
    duck(buffer.duration);
  }
}

function duck(duration) {
  const now = audioContext.currentTime;
  const gain = passthroughGain.gain;
  gain.cancelScheduledValues(now);
  gain.setValueAtTime(gain.value, now);
  gain.linearRampToValueAtTime(settings.originalGain, now + 0.08);
  gain.setValueAtTime(settings.originalGain, now + duration);
  gain.linearRampToValueAtTime(1.0, now + duration + 0.25);
}

function applyMixMode() {
  if (!passthroughGain) return;
  const mode = settings.originalAudio;
  passthroughGain.gain.value =
    mode === 'translation_only' ? 0.0 : mode === 'mixed' ? settings.originalGain : 1.0;
}

// ------------------------------------------------------------------ capture

async function start(streamId, incomingSettings) {
  settings = incomingSettings;
  running = true;
  reconnectAttempts = 0;
  playheadTime = 0;
  pendingFrames = [];

  report('ws:state', { state: 'connecting' });
  sessionId = await createSession();
  report('session:created', { sessionId });

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  // Creating the context at the target rate lets the browser resample for us.
  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  playheadTime = audioContext.currentTime;
  sourceNode = audioContext.createMediaStreamSource(mediaStream);

  // Without this the user hears nothing: capturing a tab detaches its audio
  // from the speakers.
  passthroughGain = audioContext.createGain();
  applyMixMode();
  sourceNode.connect(passthroughGain).connect(audioContext.destination);

  await audioContext.audioWorklet.addModule(chrome.runtime.getURL('offscreen/capture-worklet.js'));
  captureNode = new AudioWorkletNode(audioContext, 'voicebridge-capture');
  captureNode.port.onmessage = (event) => {
    const frame = event.data;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(frame);
    } else if (running) {
      pendingFrames.push(frame);
      if (pendingFrames.length > MAX_PENDING_FRAMES) pendingFrames.shift();
    }
  };
  sourceNode.connect(captureNode);
  // A worklet with no downstream connection may be garbage collected; route it
  // to a silent sink so it keeps pulling audio without being heard twice.
  const sink = audioContext.createGain();
  sink.gain.value = 0;
  captureNode.connect(sink).connect(audioContext.destination);

  openSocket();
  report('capture:started', {});
}

async function stop() {
  running = false;                      // before closing, so no reconnect fires
  clearTimeout(reconnectTimer); reconnectTimer = null;
  clearInterval(pingTimer); pingTimer = null;
  reconnectAttempts = 0;
  pendingFrames = [];

  try { send({ type: CLIENT.AUDIO_STOP }); } catch {}
  try { if (captureNode) captureNode.disconnect(); } catch {}
  try { if (sourceNode) sourceNode.disconnect(); } catch {}
  try { if (passthroughGain) passthroughGain.disconnect(); } catch {}
  try { if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop()); } catch {}
  try { if (audioContext) await audioContext.close(); } catch {}
  try { if (socket && socket.readyState <= 1) socket.close(); } catch {}

  if (sessionId) {
    const base = settings?.serverUrl?.replace(/\/+$/, '');
    const headers = settings?.apiToken ? { authorization: `Bearer ${settings.apiToken}` } : {};
    try { await fetch(`${base}/v1/sessions/${sessionId}`, { method: 'DELETE', headers }); } catch {}
  }

  audioContext = null; mediaStream = null; sourceNode = null;
  captureNode = null; passthroughGain = null; socket = null; sessionId = null;
  report('capture:stopped', {});
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target !== 'offscreen') return undefined;
  (async () => {
    try {
      if (message.type === 'start') {
        await start(message.streamId, message.settings);
        sendResponse({ ok: true });
      } else if (message.type === 'stop') {
        await stop();
        sendResponse({ ok: true });
      } else if (message.type === 'pause') {
        send({ type: CLIENT.PAUSE });
        sendResponse({ ok: true });
      } else if (message.type === 'resume') {
        send({ type: CLIENT.RESUME });
        sendResponse({ ok: true });
      } else if (message.type === 'settings') {
        settings = { ...settings, ...message.settings };
        applyMixMode();
        sendResponse({ ok: true });
      } else {
        sendResponse({ ok: false, error: 'unknown message' });
      }
    } catch (err) {
      console.error('[voicebridge offscreen]', err);
      report('backend:error', { message: String(err.message || err) });
      sendResponse({ ok: false, error: String(err.message || err) });
    }
  })();
  return true; // async sendResponse
});
