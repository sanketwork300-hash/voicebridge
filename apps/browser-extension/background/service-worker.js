// VoiceBridge background service worker.
//
// Coordinates popup <-> offscreen (capture + WebSocket) <-> content (overlay).
// It holds no audio and no socket: MV3 terminates an idle service worker, so
// anything long-lived lives in the offscreen document instead.

import { DEFAULT_SETTINGS } from '../common/protocol.js';

const OFFSCREEN_PATH = 'offscreen/offscreen.html';

const state = {
  capturing: false,
  paused: false,
  tabId: null,
  wsState: 'idle',       // idle | connecting | connected | reconnecting | closed | error
  sessionId: null,
  language: null,
  lastError: null,
  lastWarning: null,
  metrics: {},
};

// ------------------------------------------------------------------ settings

async function getSettings() {
  const stored = await chrome.storage.local.get('settings');
  return { ...DEFAULT_SETTINGS, ...(stored.settings || {}) };
}

async function saveSettings(patch) {
  const next = { ...(await getSettings()), ...patch };
  await chrome.storage.local.set({ settings: next });
  // Live-apply what can change mid-session (mix mode, subtitle styling).
  if (state.capturing) {
    send({ target: 'offscreen', type: 'settings', settings: next });
    if (state.tabId != null) {
      tell(state.tabId, { type: 'overlay:settings', settings: next });
    }
  }
  return next;
}

// ----------------------------------------------------------------- helpers

function send(message) {
  return chrome.runtime.sendMessage(message).catch(() => {});
}

function tell(tabId, message) {
  return chrome.tabs.sendMessage(tabId, message).catch(() => {});
}

async function hasOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_PATH)],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_PATH,
    reasons: ['USER_MEDIA'],
    justification: 'Capture tab audio and play translated speech for live translation.',
  });
}

async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: 'ping' });
    return true;
  } catch {
    // Not injected yet — normal for tabs opened before the extension loaded.
  }
  try {
    await chrome.scripting.insertCSS({ target: { tabId }, files: ['content/overlay.css'] });
    await chrome.scripting.executeScript({ target: { tabId }, files: ['content/overlay.js'] });
    return true;
  } catch (err) {
    // Restricted pages (chrome://, the Web Store) cannot be scripted. Capture
    // still works; only the on-page overlay is unavailable.
    console.warn('[voicebridge] overlay unavailable on this page:', err?.message);
    return false;
  }
}

// ------------------------------------------------------------------ capture

async function startCapture(tabId) {
  const settings = await getSettings();
  await ensureOffscreen();
  const overlayReady = await ensureContentScript(tabId);

  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });

  const response = await chrome.runtime.sendMessage({
    target: 'offscreen',
    type: 'start',
    streamId,
    settings,
  });
  if (!response?.ok) {
    throw new Error(response?.error || 'offscreen document failed to start capture');
  }

  state.capturing = true;
  state.paused = false;
  state.tabId = tabId;
  state.lastError = null;
  await chrome.storage.local.set({ capturing: true, tabId });
  await chrome.action.setBadgeText({ text: 'ON' });
  await chrome.action.setBadgeBackgroundColor({ color: '#3ecf8e' });

  if (overlayReady) {
    await tell(tabId, { type: 'overlay:mount', settings });
  }
}

async function stopCapture() {
  if (await hasOffscreen()) {
    await chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop' }).catch(() => {});
  }
  if (state.tabId != null) {
    await tell(state.tabId, { type: 'overlay:unmount' });
  }
  state.capturing = false;
  state.paused = false;
  state.tabId = null;
  state.sessionId = null;
  state.wsState = 'idle';
  await chrome.storage.local.set({ capturing: false, tabId: null });
  await chrome.action.setBadgeText({ text: '' });
}

// ----------------------------------------------------------------- messages

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target && message.target !== 'background') return undefined;

  (async () => {
    try {
      switch (message.type) {
        case 'capture:start': {
          const tab = message.tabId
            ? await chrome.tabs.get(message.tabId)
            : (await chrome.tabs.query({ active: true, currentWindow: true }))[0];
          if (!tab) throw new Error('no active tab to capture');
          await startCapture(tab.id);
          sendResponse({ ok: true, state });
          break;
        }
        case 'capture:stop':
          await stopCapture();
          sendResponse({ ok: true, state });
          break;
        case 'capture:pause':
          send({ target: 'offscreen', type: 'pause' });
          state.paused = true;
          if (state.tabId != null) tell(state.tabId, { type: 'overlay:paused', paused: true });
          sendResponse({ ok: true, state });
          break;
        case 'capture:resume':
          send({ target: 'offscreen', type: 'resume' });
          state.paused = false;
          if (state.tabId != null) tell(state.tabId, { type: 'overlay:paused', paused: false });
          sendResponse({ ok: true, state });
          break;
        case 'capture:status':
          sendResponse({ ok: true, state, settings: await getSettings() });
          break;
        case 'settings:get':
          sendResponse({ ok: true, settings: await getSettings() });
          break;
        case 'settings:set':
          sendResponse({ ok: true, settings: await saveSettings(message.settings || {}) });
          break;

        // ---- events forwarded from the offscreen document ----
        case 'ws:state':
          state.wsState = message.state;
          if (message.state === 'reconnecting' && state.tabId != null) {
            tell(state.tabId, {
              type: 'overlay:status',
              message: `Reconnecting (attempt ${message.attempt})…`,
            });
          }
          break;
        case 'session:created':
          state.sessionId = message.sessionId;
          break;
        case 'session:ended':
          state.metrics = message.metrics || {};
          break;
        case 'language':
          state.language = message.language;
          break;
        case 'subtitle':
          if (state.tabId != null) tell(state.tabId, { type: 'overlay:subtitle', ...message });
          break;
        case 'backend:warning':
          state.lastWarning = message.message;
          if (state.tabId != null) {
            tell(state.tabId, { type: 'overlay:status', message: message.message });
          }
          break;
        case 'backend:error':
          state.lastError = message.message;
          if (state.tabId != null) {
            tell(state.tabId, { type: 'overlay:error', message: message.message });
          }
          break;
        case 'capture:started':
        case 'capture:stopped':
          break;
        default:
          sendResponse({ ok: false, error: `unknown message ${message.type}` });
          return;
      }
      if (!['capture:start', 'capture:stop', 'capture:status', 'settings:get',
            'settings:set', 'capture:pause', 'capture:resume'].includes(message.type)) {
        sendResponse({ ok: true });
      }
    } catch (err) {
      console.error('[voicebridge background]', err);
      sendResponse({ ok: false, error: String(err?.message || err) });
    }
  })();

  return true; // keep the channel open for the async response
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  if (tabId === state.tabId) await stopCapture();
});

chrome.runtime.onStartup.addListener(async () => {
  // A browser restart never leaves a live capture; clear stale UI state.
  await chrome.storage.local.set({ capturing: false, tabId: null });
  await chrome.action.setBadgeText({ text: '' });
});
