// VoiceBridge subtitle overlay (content script).
//
// Renders committed translations, optional dual (original + translated) lines,
// and provisional partial text that is visually distinct from committed text.
// The distinction is the point: partial recognition can be retracted, and a
// viewer must be able to tell "the system is still deciding" from "this is what
// was said".
//
// The overlay is inert (pointer-events: none) and lives in a single fixed
// container, so it never interferes with the host page's controls.

(() => {
  if (window.__voicebridgeOverlayLoaded) return;
  window.__voicebridgeOverlayLoaded = true;

  const MAX_LINES_DEFAULT = 2;
  const PARTIAL_TIMEOUT_MS = 4000;

  let root = null;
  let settings = null;
  let cues = [];               // committed cues, newest last
  let partialText = '';
  let partialTimer = null;
  let statusTimer = null;

  function ensureRoot() {
    if (root && document.body.contains(root)) return root;
    root = document.createElement('div');
    root.id = 'voicebridge-overlay';
    root.setAttribute('role', 'region');
    root.setAttribute('aria-label', 'VoiceBridge live translated subtitles');
    root.setAttribute('aria-live', 'polite');
    root.setAttribute('aria-atomic', 'false');
    document.documentElement.appendChild(root);
    applySettings();
    return root;
  }

  function applySettings() {
    if (!root || !settings) return;
    root.dataset.position = settings.subtitlePosition || 'bottom';
    root.style.setProperty('--vb-size', `${settings.subtitleSize || 22}px`);
    root.style.setProperty('--vb-opacity', String(settings.subtitleOpacity ?? 0.85));
  }

  function render() {
    const host = ensureRoot();
    host.textContent = '';

    const maxLines = settings?.maxLines || MAX_LINES_DEFAULT;
    for (const cue of cues.slice(-maxLines)) {
      if (settings?.dualSubtitles && cue.source) {
        host.appendChild(line('vb-cue vb-original', cue.source, cue.speaker));
      }
      host.appendChild(line('vb-cue vb-translation', cue.translation, cue.speaker));
    }

    if (partialText) {
      host.appendChild(line('vb-cue vb-translation vb-partial', partialText));
    }
  }

  function line(className, text, speaker) {
    const el = document.createElement('div');
    el.className = className;
    if (speaker !== undefined && speaker !== null) {
      const tag = document.createElement('span');
      tag.className = 'vb-speaker';
      tag.textContent = `Speaker ${speaker}`;
      el.appendChild(tag);
    }
    // textContent, never innerHTML: subtitle text is remote content and must
    // never be able to inject markup into the host page.
    el.appendChild(document.createTextNode(text));
    return el;
  }

  function showBanner(className, message, ttl = 6000) {
    const host = ensureRoot();
    const existing = host.querySelector(`.${className}`);
    if (existing) existing.remove();
    const el = document.createElement('div');
    el.className = `vb-cue ${className}`;
    el.textContent = message;
    host.appendChild(el);
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => { el.remove(); }, ttl);
  }

  function mount(incoming) {
    settings = incoming;
    cues = [];
    partialText = '';
    ensureRoot();
    render();
  }

  function unmount() {
    clearTimeout(partialTimer);
    clearTimeout(statusTimer);
    if (root) root.remove();
    root = null;
    cues = [];
    partialText = '';
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    switch (message.type) {
      case 'ping':
        sendResponse({ ok: true });
        return true;

      case 'overlay:mount':
        mount(message.settings);
        break;

      case 'overlay:unmount':
        unmount();
        break;

      case 'overlay:settings':
        settings = { ...settings, ...message.settings };
        applySettings();
        render();
        break;

      case 'overlay:paused':
        if (message.paused) showBanner('vb-status', 'Translation paused', 3000);
        break;

      case 'overlay:subtitle': {
        if (message.kind === 'partial') {
          partialText = message.source || '';
          clearTimeout(partialTimer);
          // A partial that stops updating means the recogniser moved on
          // without committing; drop it rather than leave it on screen.
          partialTimer = setTimeout(() => { partialText = ''; render(); }, PARTIAL_TIMEOUT_MS);
          render();
        } else if (message.kind === 'translation') {
          partialText = '';
          clearTimeout(partialTimer);
          cues.push({
            source: message.source,
            translation: message.translation,
            speaker: message.speaker,
          });
          if (cues.length > 20) cues = cues.slice(-20);
          render();
        }
        break;
      }

      case 'overlay:status':
        showBanner('vb-status', message.message);
        break;

      case 'overlay:error':
        showBanner('vb-error', message.message, 10000);
        break;

      default:
        break;
    }
    return undefined;
  });
})();
