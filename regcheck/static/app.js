/* Reg Check — inspection bay front-end.
   Talks to the Flask API, streams events over SSE, and renders each verified
   vehicle as a card headed by its rendered number plate (the header links to the
   listing). Dynamic text is escaped before it reaches the DOM. */

'use strict';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const MODES = {
  fast:     { max_images: 12, early_stop_votes: 3, min_images: 6,
              hint: 'Reads ~12 photos, stops at a 3-photo match. Quickest sweep.' },
  balanced: { max_images: 24, early_stop_votes: 4, min_images: 10,
              hint: 'Reads up to 24 photos, plate shots first. Good default.' },
  thorough: { max_images: 40, early_stop_votes: 0, min_images: 40,
              hint: 'Reads every photo with no early stop. Slowest, most complete.' },
};
let mode = 'balanced';

let es = null;        // EventSource
let seenN = -1;       // highest event index rendered (dedupes reconnect replay)
let tallies = { verified: 0, dealer: 0, review: 0 };

/* ---- theme ------------------------------------------------------------- */
function initTheme() {
  const stored = localStorage.getItem('regcheck-theme');
  const t = stored || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  document.documentElement.setAttribute('data-theme', t);
}
$('#themeToggle').addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('regcheck-theme', next);
});

/* ---- credentials ------------------------------------------------------- */
async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    const d = await r.json();
    setConn(d.configured);
  } catch { setConn(false); }
}
function setConn(ok) {
  const pill = $('#conn');
  pill.dataset.ok = ok ? 'true' : 'false';
  pill.textContent = ok ? 'connected' : 'not set';
  $('#credHint').innerHTML = ok
    ? 'Connected. Every plate is verified against DVSA and its full MOT history recorded.'
    : 'Not set — the tool still reads plates, but records the top guess <b>unverified</b> with no MOT history. Enter your four DVSA credentials to verify.';
}
$('#credToggle').addEventListener('click', (e) => {
  const form = $('#credForm');
  const open = form.hidden;
  form.hidden = !open;
  e.target.setAttribute('aria-expanded', String(open));
});
$('#saveCred').addEventListener('click', async () => {
  const body = {};
  for (const f of ['MOT_CLIENT_ID', 'MOT_CLIENT_SECRET', 'MOT_API_KEY', 'MOT_TOKEN_URL']) {
    body[f] = $('#' + f).value;
  }
  const btn = $('#saveCred');
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    const r = await fetch('/api/config', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    setConn(d.configured);
    for (const f of Object.keys(body)) $('#' + f).value = '';
    $('#credForm').hidden = true;
    $('#credToggle').setAttribute('aria-expanded', 'false');
  } finally { btn.disabled = false; btn.textContent = 'Save credentials'; }
});

/* ---- thoroughness ------------------------------------------------------ */
function setMode(m) {
  mode = m;
  document.querySelectorAll('#thoroughness button').forEach((b) =>
    b.setAttribute('aria-checked', String(b.dataset.mode === m)));
  $('#modeHint').textContent = MODES[m].hint;
}
document.querySelectorAll('#thoroughness button').forEach((b) =>
  b.addEventListener('click', () => setMode(b.dataset.mode)));

/* ---- run control ------------------------------------------------------- */
$('#startBtn').addEventListener('click', start);
$('#stopBtn').addEventListener('click', async () => {
  $('#stopBtn').disabled = true;
  await fetch('/api/stop', { method: 'POST' });
});

async function start() {
  const urls = $('#urls').value.trim();
  if (!urls) { flashStatus('Paste at least one link', 'error'); return; }
  const { max_images, early_stop_votes, min_images } = MODES[mode];
  resetView();
  setRunning(true);
  try {
    const r = await fetch('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls, options: { max_images, early_stop_votes, min_images } }),
    });
    const d = await r.json();
    if (!r.ok) { flashStatus(d.error || 'Could not start', 'error'); setRunning(false); return; }
    attachStream();
  } catch (e) { flashStatus('Server not reachable', 'error'); setRunning(false); }
}

function setRunning(on) {
  $('#startBtn').disabled = on;
  $('#stopBtn').hidden = !on;
  $('#stopBtn').disabled = false;
  $('#progressWrap').hidden = !on;
  setLamp(on ? 'running' : 'idle');
  $('#statusText').textContent = on ? 'Inspecting…' : 'Idle';
}
function setLamp(state) { $('#statusLamp').dataset.state = state; }
function flashStatus(text, state) { $('#statusText').textContent = text; if (state) setLamp(state); }

function resetView() {
  seenN = -1;
  tallies = { verified: 0, dealer: 0, review: 0 };
  $('#cards').innerHTML = '';
  $('#dealerList').innerHTML = ''; $('#reviewList').innerHTML = '';
  $('#log').textContent = '';
  $('#empty').hidden = true;
  $('#shelves').hidden = true;
  $('#dealerShelf').hidden = true; $('#reviewShelf').hidden = true;
  $('#tallies').hidden = false;
  updateTallies();
  $('#progressBar').style.width = '0%';
  $('#progressLabel').textContent = '';
  $('#shortlistSection').hidden = true;
  updateReviewAvailability(0);
}

/* ---- SSE --------------------------------------------------------------- */
function attachStream() {
  if (es) es.close();
  es = new EventSource('/api/stream');
  es.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch { return; }
    if (typeof ev._n === 'number') { if (ev._n <= seenN) return; seenN = ev._n; }
    handle(ev);
  };
  es.onerror = () => { /* browser auto-reconnects; keepalives hold it open */ };
}

function handle(ev) {
  switch (ev.type) {
    case 'log': appendLog(ev.text); break;
    case 'status': $('#statusText').textContent = ev.text; break;
    case 'progress': setProgress(ev.current, ev.total); break;
    case 'result': tallies.verified++; updateTallies(); renderResult(ev.result); break;
    case 'review': tallies[ev.category === 'dealer' ? 'dealer' : 'review']++;
      updateTallies(); renderReview(ev); break;
    case 'done': onDone(ev.summary); break;
  }
}

function onDone(s) {
  setRunning(false);
  $('#stopBtn').hidden = true;
  $('#progressWrap').hidden = true;
  if (s && s.stopped) { flashStatus('Stopped', 'idle'); }
  else { setLamp('done'); $('#statusText').textContent =
    `Done — ${s ? s.verified : 0} verified`; }
  if (es) { es.close(); es = null; }
  if (!$('#cards').children.length && $('#dealerShelf').hidden && $('#reviewShelf').hidden) {
    $('#empty').hidden = false;
  }
  updateReviewAvailability(tallies.verified);
  // Auto-run the AI review unless the user wants to fix manual-review plates first.
  if (!(s && s.stopped) && tallies.verified > 0 && !$('#llmWait').checked) {
    setTimeout(startReview, 500);
  }
}

function setProgress(cur, total) {
  const pct = total ? Math.round((cur / total) * 100) : 0;
  $('#progressBar').style.width = pct + '%';
  $('#progressLabel').textContent = `${cur} / ${total}`;
}
function appendLog(text) {
  const log = $('#log');
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  log.textContent += text + '\n';
  if (atBottom) log.scrollTop = log.scrollHeight;
}
function updateTallies() {
  $('#tallyVerified').textContent = tallies.verified;
  $('#tallyDealer').textContent = tallies.dealer;
  $('#tallyReview').textContent = tallies.review;
}

/* ---- rendering --------------------------------------------------------- */
function plateHTML(reg, front) {
  return `<span class="plate${front ? ' plate--front' : ''}">
    <span class="plate__gb">UK</span><span class="plate__reg">${esc(reg)}</span></span>`;
}

function renderResult(r) {
  const front = !r.verified;               // white front plate = unverified
  const badges = [];
  if (r.verified && r.tier === 3) badges.push('<span class="badge badge--verified">✓ verified</span>');
  else if (r.verified && r.tier === 2) badges.push('<span class="badge badge--verified">✓ verified · model n/a</span>');
  else if (r.verified && r.tier === 1) badges.push('<span class="badge badge--warn">model differs</span>');
  if (r.corrected) badges.push('<span class="badge badge--corrected">OCR-corrected</span>');
  if (!r.verified) badges.push('<span class="badge badge--unverified">unverified</span>');

  const veh = r.verified
    ? `<b>${esc(r.make)} ${esc(r.model)}</b>` +
      join([r.colour, r.fuel, r.firstUsed && 'first used ' + r.firstUsed])
    : `<b>${esc(r.make)} ${esc(r.model)}</b><span class="dot">·</span>listing details (not DVSA-verified)`;

  const readouts = [];
  if (r.latestMileage) readouts.push(`<span>latest <b>${esc(r.latestMileage)}</b></span>`);
  if (r.motExpiry) readouts.push(`<span>MOT to <b>${esc(r.motExpiry)}</b></span>`);
  if (r.latestResult) readouts.push(`<span>last test <b>${esc(r.latestResult)}</b></span>`);

  const card = document.createElement('div');
  card.className = 'card' + (r.verified ? ' card--flash' : '');
  card.innerHTML = `
    <a class="card__head" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">
      ${plateHTML(r.plate, front)}
      <span class="card__headmeta">
        <span class="card__where">${esc(r.location || 'location n/a')}</span>
        <span class="card__site">${esc(r.site)} ↗</span>
      </span>
      <span class="card__price">${esc(r.price || 'N/A')}</span>
      <span class="card__badges">${badges.join('')}</span>
    </a>
    <div class="card__body">
      <div class="vehline">${veh}</div>
      ${readouts.length ? `<div class="readouts">${readouts.join('')}</div>` : ''}
      ${r.warning ? `<div class="warn-note">${esc(r.warning)}</div>` : ''}
      ${stripHTML(r)}
      ${historyHTML(r)}
    </div>`;
  $('#cards').prepend(card);
}

function join(parts) {
  const p = parts.filter(Boolean).map(esc);
  return p.length ? `<span class="dot">·</span>${p.join('<span class="dot">·</span>')}` : '';
}

function stripHTML(r) {
  if (!r.tests || !r.tests.length) return '';
  const segs = r.tests.map((t) =>
    `<span class="strip__seg strip__seg--${t.result === 'PASSED' ? 'pass' : t.result === 'FAILED' ? 'fail' : ''}" title="${esc(t.date)} ${esc(t.result)}"></span>`).join('');
  const cap = `${r.passes || 0} pass · ${r.fails || 0} fail`;
  return `<div class="strip"><div class="strip__row">${segs}<span class="strip__cap">${cap}</span></div></div>`;
}

function historyHTML(r) {
  if (!r.tests || !r.tests.length) return '';
  const tests = r.tests.slice().reverse().map((t) => {
    const defects = (t.defects || []).map((d) => {
      const type = d.dangerous ? 'DANGEROUS' : (d.type || '');
      const danger = d.dangerous ? ' <span class="danger">[DANGEROUS]</span>' : '';
      return `<li class="defect" data-t="${esc(type)}">${esc(d.type)}: ${esc(d.text)}${danger}</li>`;
    }).join('');
    return `<div class="test">
      <div class="test__line">
        <span>${esc(t.date)}</span>
        <span class="test__result" data-r="${esc(t.result)}">${esc(t.result)}</span>
        <span class="test__mileage">${esc(t.mileage || '')}</span>
      </div>${defects ? `<ul class="defects">${defects}</ul>` : ''}</div>`;
  }).join('');
  return `<details class="history"><summary>MOT history · ${r.tests.length} test(s)</summary>${tests}</details>`;
}

function renderReview(ev) {
  const isDealer = ev.category === 'dealer';
  $(isDealer ? '#dealerShelf' : '#reviewShelf').hidden = false;
  $('#shelves').hidden = false;
  $(isDealer ? '#dealerCount' : '#reviewCount').textContent =
    isDealer ? tallies.dealer : tallies.review;
  const item = document.createElement('div');
  item.className = 'shelf__item';
  const meta =
    (ev.location && ev.location !== 'N/A' ? `<span class="note">${esc(ev.location)}</span>` : '') +
    (ev.price ? `<span class="note">${esc(ev.price)}</span>` : '') +
    (ev.note ? `<span class="note">${esc(ev.note)}</span>` : '');
  item.innerHTML =
    `<div class="shelf__row">
       <a href="${esc(ev.url)}" target="_blank" rel="noopener noreferrer">${esc(ev.url)}</a>` +
    (isDealer ? '' : `<button class="pen" type="button" title="Read the reg yourself and enter it" aria-label="Enter reg">✎</button>`) +
    `</div>${meta}` +
    (isDealer ? '' :
      `<form class="reg-edit" hidden>
         <input class="reg-input" placeholder="type the reg, e.g. AB12 CDE" spellcheck="false" maxlength="10" />
         <button class="btn btn--ghost" type="submit">Check</button>
         <span class="reg-msg"></span>
       </form>`);
  if (!isDealer) wireManualEdit(item, ev);
  $(isDealer ? '#dealerList' : '#reviewList').appendChild(item);
}

function wireManualEdit(item, ev) {
  const pen = item.querySelector('.pen');
  const form = item.querySelector('.reg-edit');
  const input = item.querySelector('.reg-input');
  const msg = item.querySelector('.reg-msg');
  pen.addEventListener('click', () => {
    form.hidden = !form.hidden;
    if (!form.hidden) input.focus();
  });
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const plate = input.value.trim();
    if (!plate) return;
    msg.textContent = 'Checking…';
    try {
      const r = await fetch('/api/lookup', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plate, url: ev.url, location: ev.location,
                               price: ev.price, site: ev.site }),
      });
      const d = await r.json();
      if (!r.ok) { msg.textContent = d.error || 'Not found'; return; }
      renderResult(d.result);
      tallies.verified++; tallies.review = Math.max(0, tallies.review - 1);
      updateTallies(); updateReviewAvailability(tallies.verified);
      $('#reviewCount').textContent = tallies.review;
      item.remove();
      if (!$('#reviewList').children.length) $('#reviewShelf').hidden = true;
    } catch { msg.textContent = 'Server error'; }
  });
}

/* ---- AI review (local LLM tournament) ---------------------------------- */
let reviewEs = null;

async function loadLlmConfig() {
  try {
    const d = await (await fetch('/api/llm-config')).json();
    if (!$('#llmModel').value) $('#llmModel').value = d.model || '';
    if (!$('#llmBaseUrl').value) $('#llmBaseUrl').value = d.base_url || '';
    updateReviewAvailability(d.results || 0);
    if (d.running) { setReviewRunning(true); attachReviewStream(); }
  } catch { /* server not up yet */ }
}

function updateReviewAvailability(n) {
  const pill = $('#reviewCountPill');
  pill.textContent = `${n} vehicle${n === 1 ? '' : 's'}`;
  pill.dataset.ok = n > 0 ? 'true' : 'false';
  $('#reviewBtn').disabled = n < 1;
}

$('#llmAdvToggle').addEventListener('click', (e) => {
  const f = $('#llmUrlField');
  f.hidden = !f.hidden;
  e.target.setAttribute('aria-expanded', String(!f.hidden));
});

const waitEl = $('#llmWait');
waitEl.checked = localStorage.getItem('regcheck-wait') === '1';
waitEl.addEventListener('change', () =>
  localStorage.setItem('regcheck-wait', waitEl.checked ? '1' : '0'));

$('#reviewBtn').addEventListener('click', startReview);

async function startReview() {
  setReviewRunning(true);
  $('#reviewStatus').textContent = 'Reviewing every verified vehicle…';
  try {
    const r = await fetch('/api/review', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: $('#llmModel').value.trim(),
                             base_url: $('#llmBaseUrl').value.trim(),
                             brief: $('#llmBrief').value.trim(),
                             shortlist: parseInt($('#llmShortlist').value, 10) || 10 }),
    });
    const d = await r.json();
    if (!r.ok) { $('#reviewStatus').textContent = d.error || 'Could not start review'; setReviewRunning(false); return; }
    attachReviewStream();
  } catch { $('#reviewStatus').textContent = 'Server not reachable'; setReviewRunning(false); }
}

function setReviewRunning(on) {
  $('#reviewBtn').disabled = on;
  $('#reviewBtn').textContent = on ? 'Reviewing…' : 'Shortlist vehicles';
}

function attachReviewStream() {
  if (reviewEs) reviewEs.close();
  reviewEs = new EventSource('/api/review/stream');
  reviewEs.onmessage = (m) => {
    let ev; try { ev = JSON.parse(m.data); } catch { return; }
    if (ev.type === 'log') { appendLog(ev.text); $('#reviewStatus').textContent = ev.text.replace(/^\[Review\]\s*/, ''); }
    else if (ev.type === 'shortlist') renderShortlist(ev);
    else if (ev.type === 'done') { setReviewRunning(false); $('#reviewStatus').textContent = 'Shortlist ready.'; if (reviewEs) { reviewEs.close(); reviewEs = null; } }
  };
  reviewEs.onerror = () => {};
}

function renderShortlist(ev) {
  if (ev.error) { $('#reviewStatus').textContent = 'Review error: ' + ev.error; return; }
  const cards = $('#shortlistCards');
  cards.innerHTML = '';
  (ev.shortlist || []).forEach((r) => cards.appendChild(shortlistCard(r)));
  $('#shortlistModel').textContent = ev.model ? `· ${ev.model}` : '';
  renderLeaderboard(ev.leaderboard || []);
  $('#shortlistSection').hidden = false;
  $('#shortlistSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function shortlistCard(r) {
  const el = document.createElement('div');
  el.className = 'sl-card' + (r.best ? ' sl-card--best' : '');
  const meta = [r.year, r.mileage, r.location].filter(Boolean).map(esc).join(' · ');
  el.innerHTML = `
    <div class="sl-card__head">
      <span class="sl-rank">${r.best ? 'BEST BUY' : '#' + r.rank}</span>
      <a class="sl-plate" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">
        ${plateHTML(r.plate, false)}</a>
      <span class="sl-price">${esc(r.price || 'N/A')}</span>
    </div>
    <div class="sl-meta">${meta}</div>
    ${r.verdict ? `<p class="sl-verdict">${esc(r.verdict)}</p>` : ''}
    <div class="sl-pc">
      <ul class="sl-pros">${(r.pros || []).map((p) => `<li>${esc(p)}</li>`).join('')}</ul>
      <ul class="sl-cons">${(r.cons || []).map((c) => `<li>${esc(c)}</li>`).join('')}</ul>
    </div>`;
  return el;
}

function renderLeaderboard(rows) {
  if (!rows.length) { $('#leaderboard').hidden = true; return; }
  $('#leaderCount').textContent = rows.length;
  const body = rows.map((r) => `<tr>
    <td class="lb-rank">${r.rank}</td>
    <td><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.plate)}</a></td>
    <td class="lb-num">${esc(r.price || '')}</td>
    <td class="lb-num">${esc(String(r.year || ''))}</td>
    <td class="lb-num">${esc(r.mileage || '')}</td>
    <td>${esc(r.location || '')}</td>
    <td class="lb-note">${esc(r.note || '')}</td>
    <td class="lb-num">${esc(String(r.rating))}</td></tr>`).join('');
  $('#leaderTable').innerHTML =
    `<thead><tr><th>#</th><th>Reg</th><th>Price</th><th>Year</th><th>Mileage</th>` +
    `<th>Where</th><th>MOT notes</th><th>Elo</th></tr></thead><tbody>${body}</tbody>`;
  $('#leaderboard').hidden = false;
}

/* ---- boot -------------------------------------------------------------- */
initTheme();
setMode('balanced');
loadConfig();
loadLlmConfig();
fetch('/api/state').then((r) => r.json()).then((d) => {
  if (d.running) { setRunning(true); attachStream(); }
}).catch(() => {});
