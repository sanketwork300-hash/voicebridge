// Shared protocol constants and helpers.
// Mirrors voicebridge/protocols/websocket/messages.py — keep both in sync.

export const CLIENT = {
  SESSION_CONFIG: 'SESSION_CONFIG',
  AUDIO_START: 'AUDIO_START',
  AUDIO_STOP: 'AUDIO_STOP',
  PAUSE: 'PAUSE',
  RESUME: 'RESUME',
  PING: 'PING',
};

export const SERVER = {
  SESSION_STARTED: 'SESSION_STARTED',
  ASR_PARTIAL: 'ASR_PARTIAL',
  ASR_STABLE: 'ASR_STABLE',
  TRANSLATION_PARTIAL: 'TRANSLATION_PARTIAL',
  TRANSLATION_FINAL: 'TRANSLATION_FINAL',
  TTS_AUDIO: 'TTS_AUDIO',
  LANGUAGE_DETECTED: 'LANGUAGE_DETECTED',
  SPEAKER_CHANGED: 'SPEAKER_CHANGED',
  WARNING: 'WARNING',
  ERROR: 'ERROR',
  SESSION_ENDED: 'SESSION_ENDED',
  PONG: 'PONG',
};

// The internal audio contract, matching the backend.
export const SAMPLE_RATE = 16000;
export const CHANNELS = 1;

export const DEFAULT_SETTINGS = {
  serverUrl: 'http://127.0.0.1:8000',
  apiToken: '',
  sourceLanguage: 'auto',
  targetLanguage: 'en',
  mode: 'subtitles',
  profile: 'balanced',
  preset: '',
  dualSubtitles: true,
  originalAudio: 'original_only', // original_only | translation_only | mixed
  originalGain: 0.25,
  subtitleSize: 22,
  subtitlePosition: 'bottom',
  subtitleOpacity: 0.85,
  maxLines: 2,
  nameRendering: 'translate',
  honorifics: 'natural_english',
  glossary: [],
};

export function httpToWs(url) {
  return url.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
}

export function buildSessionPayload(s) {
  return {
    source_language: s.sourceLanguage,
    target_language: s.targetLanguage,
    mode: s.mode,
    profile: s.profile,
    preset: s.preset || undefined,
    name_rendering: s.nameRendering,
    honorifics: s.honorifics,
    input: { type: 'browser_tab', sample_rate: SAMPLE_RATE, channels: CHANNELS },
    output: {
      subtitles: s.mode !== 'speech',
      audio: s.mode !== 'subtitles',
      dual_subtitles: !!s.dualSubtitles,
      mix: s.originalAudio,
      original_gain: s.originalGain,
    },
    glossary: { terms: s.glossary || [] },
    privacy: { save_audio: false, save_transcripts: false, save_translation: false },
  };
}
