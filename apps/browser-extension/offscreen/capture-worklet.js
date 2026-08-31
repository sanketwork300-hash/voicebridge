// Downmixes tab audio to mono and emits fixed-size 16-bit PCM frames.
//
// An AudioWorklet is used rather than the ScriptProcessorNode seen in older
// reference extensions: ScriptProcessorNode is deprecated and runs on the main
// thread, where it competes with page rendering and drops audio on busy pages
// (which video pages always are).
//
// The AudioContext is created at 16 kHz, so no resampling is needed here — the
// browser's own resampler handles rate conversion when the graph is built,
// which is both higher quality and cheaper than doing it in JS.

const FRAME_SAMPLES = 1600; // 100 ms at 16 kHz

class VoiceBridgeCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Int16Array(FRAME_SAMPLES);
    this.index = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const left = input[0];
    const right = input.length > 1 ? input[1] : left;

    for (let i = 0; i < left.length; i++) {
      const mono = (left[i] + right[i]) * 0.5;
      const clamped = Math.max(-1, Math.min(1, mono));
      // Asymmetric scaling: +1.0 must not wrap to -32768.
      this.buffer[this.index++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

      if (this.index === FRAME_SAMPLES) {
        const frame = new Int16Array(this.buffer);
        this.port.postMessage(frame.buffer, [frame.buffer]);
        this.index = 0;
      }
    }
    return true;
  }
}

registerProcessor('voicebridge-capture', VoiceBridgeCapture);
