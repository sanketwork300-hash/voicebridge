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

  // Replaced elements render no children, so appending the overlay to one would
  // silently hide it. A player that full-screens a wrapper <div> (YouTube and
  // most sites) is fine; one that full-screens the bare <video> is not
  // reachable from here at all, and we leave the overlay in <html> rather than
  // move it somewhere it definitely cannot paint.
  const CANNOT_HOST_CHILDREN = new Set(['VIDEO', 'IFRAME', 'EMBED', 'OBJECT', 'IMG', 'CANVAS']);

  // A full-screened element is promoted to the browser's *top layer*, which no
  // z-index outside its subtree can paint above -- so the overlay has to
  // actually live inside it while full-screen is active.
  function desiredParent() {
    const fs = document.fullscreenElement || document.webkitFullscreenElement || null;
    if (fs && !CANNOT_HOST_CHILDREN.has(fs.tagName)) return fs;
    return document.documentElement;
  }

  function ensureRoot() {
    const parent = desiredParent();
    // Check against the parent we actually mount into. Testing document.body
    // here would never match, because the overlay is a sibling of <body>.
    if (root && root.isConnected && root.parentNode === parent) return root;
    if (!root || !root.isConnected) {
      root = document.createElement('div');
      root.id = 'voicebridge-overlay';
      root.setAttribute('role', 'region');
      root.setAttribute('aria-label', 'VoiceBridge live translated subtitles');
      root.setAttribute('aria-live', 'polite');
      root.setAttribute('aria-atomic', 'false');
    }
    // appendChild moves an existing node, so re-parenting on a full-screen
    // change keeps the rendered cues intact.
    parent.appendChild(root);
    applySettings();
    return root;
  }

  function onFullscreenChange() {
    if (!root) return;   // not mounted; nothing to re-parent
    ensureRoot();
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

  // Registered once (the whole script is guarded by __voicebridgeOverlayLoaded).
  // Capture phase, because some players stop the event bubbling to document.
  document.addEventListener('fullscreenchange', onFullscreenChange, true);
  document.addEventListener('webkitfullscreenchange', onFullscreenChange, true);

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
