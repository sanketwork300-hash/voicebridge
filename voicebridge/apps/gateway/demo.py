"""A minimal browser demo served at ``/``.

Its purpose is diagnostic: it proves the gateway, protocol, pipeline and audio
path work end to end from a normal browser, without installing an extension.
The extension in ``apps/browser-extension`` is the real client.

It captures the microphone rather than a tab, because a plain web page cannot
capture tab audio -- that requires the ``tabCapture`` extension API.
"""

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoiceBridge Demo</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1117; --fg:#e8eaf0; --dim:#9aa3b2;
          --accent:#5b9dff; --ok:#3ecf8e; --warn:#ffb454; --err:#ff6b6b;
          --panel:#181b24; --border:#272b36; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:16px 20px; border-bottom:1px solid var(--border); display:flex;
           align-items:center; gap:14px; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0; font-weight:650; letter-spacing:-0.01em; }
  .tag { font-size:11px; color:var(--dim); border:1px solid var(--border);
         padding:2px 8px; border-radius:99px; }
  main { padding:20px; max-width:900px; margin:0 auto; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; margin-bottom:16px; }
  label { display:block; font-size:12px; color:var(--dim); margin-bottom:4px; }
  select, button { font:inherit; padding:8px 12px; border-radius:8px;
                   border:1px solid var(--border); background:var(--panel); color:var(--fg); }
  button { cursor:pointer; font-weight:600; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#05070d; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .status { display:flex; gap:16px; flex-wrap:wrap; font-size:13px; margin-bottom:16px;
            padding:12px; background:var(--panel); border:1px solid var(--border); border-radius:10px; }
  .dot { width:8px; height:8px; border-radius:50%; display:inline-block;
         margin-right:6px; background:var(--dim); }
  .dot.on { background:var(--ok); } .dot.err { background:var(--err); }
  #subs { min-height:220px; padding:16px; background:var(--panel);
          border:1px solid var(--border); border-radius:10px; }
  .line { margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid var(--border); }
  .line:last-child { border-bottom:none; }
  .src { color:var(--dim); font-size:14px; }
  .tgt { font-size:17px; margin-top:3px; }
  .partial { opacity:.55; font-style:italic; }
  .meta { font-size:11px; color:var(--dim); margin-top:4px; }
  #log { margin-top:14px; font:12px ui-monospace,monospace; color:var(--dim);
         max-height:130px; overflow:auto; white-space:pre-wrap; }
  .err { color:var(--err); } .warn { color:var(--warn); }
  .notice { margin-bottom:14px; padding:12px 14px; border-radius:10px; font-size:13px;
            background:rgba(255,180,84,.12); border:1px solid var(--warn); color:var(--fg); }
  .notice code { font:12px ui-monospace,monospace; }
</style>
</head>
<body>
<header>
  <h1>VoiceBridge</h1>
  <span class="tag" id="mode">checking…</span>
  <span class="tag">microphone demo</span>
</header>
<main>
  <div class="row">
    <div><label for="src">Source</label>
      <select id="src">
        <option value="auto">Auto-detect</option>
        <option value="ja">Japanese</option><option value="ko">Korean</option>
        <option value="en" selected>English</option><option value="hi">Hindi</option>
      </select></div>
    <div><label for="tgt">Target</label>
      <select id="tgt">
        <option value="en" selected>English</option><option value="hi">Hindi</option>
        <option value="ja">Japanese</option><option value="ko">Korean</option>
      </select></div>
    <div><label for="md">Mode</label>
      <select id="md">
        <option value="subtitles" selected>Subtitles</option>
        <option value="speech">Speech</option>
        <option value="speech_and_subtitles">Speech + subtitles</option>
      </select></div>
    <div><label for="pf">Profile</label>
      <select id="pf">
        <option value="low_latency">Low latency</option>
        <option value="balanced" selected>Balanced</option>
        <option value="accurate">Accurate</option>
      </select></div>
    <button id="go" class="primary">Start translation</button>
    <button id="stop" disabled>Stop</button>
  </div>

  <div class="status">
    <span><i class="dot" id="d-conn"></i>Connected</span>
    <span><i class="dot" id="d-cap"></i>Capturing</span>
    <span><i class="dot" id="d-tr"></i>Translating</span>
    <span>Latency: <b id="lat">—</b></span>
    <span>Detected: <b id="det">—</b></span>
  </div>

  <div id="notice" class="notice" hidden></div>
  <div id="subs"><div class="meta">Press “Start translation” and allow microphone access.</div></div>
  <div id="log"></div>
</main>
<script>
const $ = id => document.getElementById(id);
let ws=null, ctx=null, node=null, stream=null, sessionId=null, playAt=0, t0=new Map();

function log(msg, cls) {
  const d=document.createElement('div'); if(cls) d.className=cls;
  d.textContent = new Date().toLocaleTimeString()+'  '+msg;
  $('log').prepend(d);
}
function dot(id,on,err){ const e=$(id); e.className='dot'+(err?' err':(on?' on':'')); }

fetch('/health').then(r=>r.json()).then(h=>{
  $('mode').textContent = h.mock_mode ? 'mock mode' : 'live models';
  if (h.mock_mode) {
    $('notice').hidden = false;
    $('notice').innerHTML =
      '<b>Mock mode.</b> The gateway is running with <code>provider: mock</code> for ASR, ' +
      'translation and TTS, so nothing you say is recognised: the transcript below is a ' +
      'fixed demo script replayed against the audio clock, and the translation is a ' +
      'phrase table. This proves the capture → WebSocket → pipeline → subtitle path only. ' +
      'For real translation install the model extras and switch the providers in ' +
      '<code>config/voicebridge.yaml</code> (see README, “Going live”).';
  }
}).catch(()=>{ $('mode').textContent='gateway unreachable'; });

async function start(){
  $('go').disabled=true;
  try { stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,
        echoCancellation:true, noiseSuppression:true}}); }
  catch(e){ log('Microphone permission denied. Allow access in the address bar and retry.','err');
            $('go').disabled=false; return; }
  dot('d-cap',true);

  const body = { source_language:$('src').value, target_language:$('tgt').value,
    mode:$('md').value, profile:$('pf').value,
    input:{type:'microphone', sample_rate:16000, channels:1},
    output:{subtitles:true, audio:$('md').value!=='subtitles'} };
  const res = await fetch('/v1/sessions',{method:'POST',headers:{'content-type':'application/json'},
                                          body:JSON.stringify(body)});
  if(!res.ok){ log('Could not create session: '+res.status,'err'); $('go').disabled=false; return; }
  const session = await res.json();
  sessionId = session.session_id;

  const proto = location.protocol==='https:'?'wss:':'ws:';
  ws = new WebSocket(`${proto}//${location.host}/v1/sessions/${sessionId}/stream`);
  ws.binaryType='arraybuffer';
  ws.onopen = ()=>{ dot('d-conn',true); log('connected — session '+sessionId.slice(0,8));
    ws.send(JSON.stringify({type:'SESSION_CONFIG', input:{sample_rate:16000, channels:1}}));
    ws.send(JSON.stringify({type:'AUDIO_START'})); };
  ws.onclose = ()=>{ dot('d-conn',false); log('disconnected'); };
  ws.onerror = ()=>{ dot('d-conn',false,true); log('websocket error','err'); };
  ws.onmessage = ev => handle(JSON.parse(ev.data));

  ctx = new AudioContext({sampleRate:16000});
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET],{type:'text/javascript'})));
  const srcNode = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx,'vb-capture');
  node.port.onmessage = e => {
    if(ws && ws.readyState===1) ws.send(e.data);
  };
  srcNode.connect(node);
  // Keep the worklet running without routing the mic to the speakers.
  const mute = ctx.createGain(); mute.gain.value=0;
  node.connect(mute).connect(ctx.destination);
  playAt = ctx.currentTime;
  $('stop').disabled=false;
}

function handle(ev){
  const t = ev.event_type;
  if(t==='ASR_PARTIAL'){ renderPartial(ev.text); }
  else if(t==='ASR_STABLE'){ renderStable(ev); }
  else if(t==='TRANSLATION_FINAL'){ dot('d-tr',true); renderTranslation(ev); }
  else if(t==='TTS_AUDIO'){ playAudio(ev); }
  else if(t==='LANGUAGE_DETECTED'){ $('det').textContent = ev.language + (ev.stable?' (stable)':''); }
  else if(t==='WARNING'){ log('warning: '+ev.message,'warn'); }
  else if(t==='ERROR'){ log('error: '+ev.message,'err'); }
  else if(t==='SESSION_ENDED'){ const m=ev.metrics||{};
    if(m.end_to_end_p50) $('lat').textContent = m.end_to_end_p50.toFixed(2)+' s'; }
}

let partialEl=null;
function renderPartial(text){
  if(!text) return;
  if(!partialEl){ partialEl=document.createElement('div'); partialEl.className='line partial';
    $('subs').prepend(partialEl); }
  partialEl.innerHTML = `<div class="src"></div><div class="meta">recognising…</div>`;
  partialEl.querySelector('.src').textContent = text;
}
function renderStable(ev){
  if(partialEl){ partialEl.remove(); partialEl=null; }
  const el=document.createElement('div'); el.className='line'; el.id='seg-'+ev.sequence_id;
  el.innerHTML = `<div class="src"></div><div class="tgt partial">translating…</div>
                  <div class="meta">${ev.start.toFixed(1)}s – ${ev.end.toFixed(1)}s</div>`;
  el.querySelector('.src').textContent = ev.text;
  $('subs').prepend(el);
  t0.set(ev.text, performance.now());
}
function renderTranslation(ev){
  let host=null;
  for(const el of $('subs').querySelectorAll('.line')){
    const s=el.querySelector('.src');
    if(s && ev.source_text.includes(s.textContent.trim())){ host=el; break; }
  }
  if(!host){ host=document.createElement('div'); host.className='line';
    host.innerHTML='<div class="src"></div><div class="tgt"></div><div class="meta"></div>';
    host.querySelector('.src').textContent = ev.source_text; $('subs').prepend(host); }
  const tgt=host.querySelector('.tgt'); tgt.className='tgt'; tgt.textContent = ev.translated_text;
  const started=t0.get(ev.source_text);
  if(started){ $('lat').textContent = ((performance.now()-started)/1000).toFixed(2)+' s'; }
  host.querySelector('.meta').textContent =
    `${ev.source_language} → ${ev.target_language} · committed`;
}

function playAudio(ev){
  if(!ctx) return;
  const raw = atob(ev.audio); const bytes = new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i);
  const pcm = new Int16Array(bytes.buffer);
  const buf = ctx.createBuffer(ev.channels||1, pcm.length, ev.sample_rate||16000);
  const ch = buf.getChannelData(0);
  for(let i=0;i<pcm.length;i++) ch[i]=pcm[i]/32768;
  const src = ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination);
  // Schedule sequentially so segments never overlap.
  playAt = Math.max(playAt, ctx.currentTime);
  src.start(playAt); playAt += buf.duration;
}

async function stop(){
  $('stop').disabled=true;
  try { if(ws && ws.readyState===1){ ws.send(JSON.stringify({type:'AUDIO_STOP'})); } } catch(e){}
  try { if(node) node.disconnect(); } catch(e){}
  try { if(stream) stream.getTracks().forEach(t=>t.stop()); } catch(e){}
  try { if(ctx) await ctx.close(); } catch(e){}
  if(sessionId){ try{ await fetch('/v1/sessions/'+sessionId,{method:'DELETE'}); }catch(e){} }
  try { if(ws) ws.close(); } catch(e){}
  ws=null; ctx=null; node=null; stream=null; sessionId=null;
  dot('d-conn',false); dot('d-cap',false); dot('d-tr',false);
  $('go').disabled=false;
}

const WORKLET = `
class VBCapture extends AudioWorkletProcessor {
  constructor(){ super(); this.buf = new Int16Array(1600); this.n = 0; }
  process(inputs){
    const ch = inputs[0] && inputs[0][0];
    if(!ch) return true;
    for(let i=0;i<ch.length;i++){
      let s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s*0x8000 : s*0x7fff;
      if(this.n === this.buf.length){
        this.port.postMessage(this.buf.slice().buffer, [this.buf.slice().buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('vb-capture', VBCapture);
`;

$('go').onclick = start;
$('stop').onclick = stop;
</script>
</body>
</html>
"""
