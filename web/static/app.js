/* ═══════════════════════════════════════════════════════════════════════════
   Nalanda Knowledge System — SPA Logic v3
   Auth (login / signup / logout) · Role-based UI · 3-D parallax
   Admin: knowledge base + user management · Chat with image display
═══════════════════════════════════════════════════════════════════════════ */

// Backend base URL — empty string = same origin (Railway full-stack).
// For Vercel frontend + separate backend: set window.API_BASE in index.html.
const API_BASE = ((window.API_BASE) || '').replace(/\/$/, '');
const u = (path) => API_BASE + path;

// ── State ─────────────────────────────────────────────────────────────────────

const S = {
  token:        localStorage.getItem('nk_token')    || '',
  role:         localStorage.getItem('nk_role')     || '',
  username:     localStorage.getItem('nk_username') || '',
  history:      [],
  jobId:        null,
  jobTimer:     null,
};

// ── DOM helpers ───────────────────────────────────────────────────────────────

const $  = id  => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// Screens
const authScreen  = $('authScreen');
const appScreen   = $('appScreen');

// Auth cards
const loginCard   = $('loginCard');
const signupCard  = $('signupCard');
const loginForm   = $('loginForm');
const signupForm  = $('signupForm');
const loginUser   = $('loginUser');
const loginPass   = $('loginPass');
const loginError  = $('loginError');
const loginBtn    = $('loginBtn');
const signupUser  = $('signupUser');
const signupEmail = $('signupEmail');
const signupPass  = $('signupPass');
const signupPassC = $('signupPassC');
const signupMsg   = $('signupMsg');
const signupBtn   = $('signupBtn');
const goSignup    = $('goSignup');
const goLogin     = $('goLogin');

// Parallax
const pBg         = $('pBg');
const pMid        = $('pMid');
const pDust       = $('pDust');
const particles   = $('particles');

// Header
const userAvatar     = $('userAvatar');
const userDisplay    = $('userDisplay');
const roleDisplay    = $('roleDisplay');
const adminToggleBtn = $('adminToggleBtn');
const logoutBtn      = $('logoutBtn');

// Admin panel
const adminPanel    = $('adminPanel');
const closePanelBtn = $('closePanelBtn');
const mainContent   = $('mainContent');

// Panel tabs
const statVectors    = $('statVectors');
const statDocs       = $('statDocs');
const refreshStatsBtn = $('refreshStatsBtn');
const uploadZone      = $('uploadZone');
const pdfFileInput    = $('pdfFileInput');
const folderInput     = $('folderInput');
const uploadStatus    = $('uploadStatus');
const uploadedPanel   = $('uploadedPanel');
const uploadedList    = $('uploadedList');
const selectAllChk    = $('selectAllChk');
const selectedCount   = $('selectedCount');
const indexSelectedBtn = $('indexSelectedBtn');
const indexProgress   = $('indexProgress');
const progressLog     = $('progressLog');
const clearIndexBtn  = $('clearIndexBtn');
const userList       = $('userList');
const refreshUsersBtn = $('refreshUsersBtn');

// Chat
const chatMessages  = $('chatMessages');
const thinkingRow   = $('thinkingRow');
const chatForm      = $('chatForm');
const chatInput     = $('chatInput');
const sendBtn       = $('sendBtn');
const topKInput     = $('topK');
const charCount     = $('charCount');
const clearChatBtn  = $('clearChatBtn');
const verifiedBadge = $('verifiedBadge');
const suggestions   = $('suggestions');

// Admin URL Sources
const urlSourceList   = $('urlSourceList');
const urlSourceInput  = $('urlSourceInput');
const urlSourceAddBtn = $('urlSourceAddBtn');
const urlSourceStatus = $('urlSourceStatus');

// Confirm dialog
const confirmOverlay = $('confirmOverlay');
const confirmTitle   = $('confirmTitle');
const confirmMsg     = $('confirmMsg');
const confirmOk      = $('confirmOk');
const confirmCancel  = $('confirmCancel');

// Templates
const msgTpl     = $('msgTpl');
const userRowTpl = $('userRowTpl');


// ════════════════════════════════════════════════════════════════════════════
//  3-D PARALLAX BACKGROUND
// ════════════════════════════════════════════════════════════════════════════

function spawnParticles(count = 22) {
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = 2 + Math.random() * 5;
    p.style.cssText = [
      `left: ${Math.random() * 100}%`,
      `width: ${size}px`,
      `height: ${size}px`,
      `animation-delay: ${(Math.random() * 12).toFixed(1)}s`,
      `animation-duration: ${(7 + Math.random() * 10).toFixed(1)}s`,
      `opacity: ${(.25 + Math.random() * .55).toFixed(2)}`,
    ].join(';');
    particles.appendChild(p);
  }
}

function initParallax() {
  spawnParticles();
  authScreen.addEventListener('mousemove', onParallaxMove);
  authScreen.addEventListener('mouseleave', onParallaxReset);
}

let _rafId = null;
let _mx = 0, _my = 0;

function onParallaxMove(e) {
  const w = authScreen.clientWidth, h = authScreen.clientHeight;
  _mx = (e.clientX / w - .5);   // -0.5 … +0.5
  _my = (e.clientY / h - .5);
  if (!_rafId) _rafId = requestAnimationFrame(applyParallax);
}

function applyParallax() {
  _rafId = null;

  // Background layer — barely moves (furthest)
  pBg.style.transform  = `translate(${_mx * -22}px, ${_my * -22}px) scale(1.12)`;
  // Mid layer — moderate
  pMid.style.transform = `translate(${_mx * -10}px, ${_my * -10}px)`;
  // Dust / vignette — opposite direction (closest)
  pDust.style.transform = `translate(${_mx * 6}px,  ${_my * 6}px)`;

  // Card 3-D tilt — subtle rotateX/Y
  const activeCard = loginCard.classList.contains('slide-out-left') ? signupCard : loginCard;
  activeCard.style.transform = `perspective(1100px) rotateY(${_mx * 9}deg) rotateX(${_my * -9}deg) translateZ(10px)`;
}

function onParallaxReset() {
  pBg.style.transform = pMid.style.transform = pDust.style.transform = '';
  loginCard.style.transform  = 'perspective(1100px) rotateX(0) rotateY(0) translateZ(0)';
  signupCard.style.transform = 'perspective(1100px) rotateX(0) rotateY(0) translateZ(0)';
}


// ════════════════════════════════════════════════════════════════════════════
//  AUTH CARD SLIDING
// ════════════════════════════════════════════════════════════════════════════

goSignup.addEventListener('click', () => {
  loginCard.classList.add('slide-out-left');
  loginCard.classList.remove('slide-in-left');
  signupCard.classList.remove('slide-out-right', 'auth-card-right');
  signupCard.classList.add('slide-in-right');
  onParallaxReset();
  setTimeout(() => signupUser.focus(), 100);
});

goLogin.addEventListener('click', () => {
  signupCard.classList.add('slide-out-right');
  signupCard.classList.remove('slide-in-right');
  loginCard.classList.remove('slide-out-left');
  loginCard.classList.add('slide-in-left');
  onParallaxReset();
  setTimeout(() => loginUser.focus(), 100);
});


// ════════════════════════════════════════════════════════════════════════════
//  SESSION HELPERS
// ════════════════════════════════════════════════════════════════════════════

function saveSession(token, role, username) {
  S.token = token; S.role = role; S.username = username;
  localStorage.setItem('nk_token',    token);
  localStorage.setItem('nk_role',     role);
  localStorage.setItem('nk_username', username);
}

function clearSession() {
  S.token = ''; S.role = ''; S.username = '';
  ['nk_token','nk_role','nk_username'].forEach(k => localStorage.removeItem(k));
}

function authHeaders(extra = {}) {
  return { 'Content-Type': 'application/json', Authorization: `Bearer ${S.token}`, ...extra };
}

async function apiPost(url, body) {
  const r = await fetch(u(url), { method: 'POST', headers: authHeaders(), body: JSON.stringify(body) });
  if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail || `HTTP ${r.status}`); }
  return r.json();
}
async function apiGet(url) {
  const r = await fetch(u(url), { headers: authHeaders() });
  if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail || `HTTP ${r.status}`); }
  return r.json();
}
async function apiPatch(url, body) {
  const r = await fetch(u(url), { method: 'PATCH', headers: authHeaders(), body: JSON.stringify(body) });
  if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail || `HTTP ${r.status}`); }
  return r.json();
}
async function apiDelete(url) {
  const r = await fetch(url, { method: 'DELETE', headers: authHeaders() });
  if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail || `HTTP ${r.status}`); }
  return r.json();
}


// ════════════════════════════════════════════════════════════════════════════
//  LOGIN
// ════════════════════════════════════════════════════════════════════════════

loginForm.addEventListener('submit', async e => {
  e.preventDefault();
  const username = loginUser.value.trim(), password = loginPass.value;
  if (!username || !password) return;

  loginBtn.disabled = true;
  loginBtn.querySelector('span').textContent = 'Entering…';
  loginError.hidden = true;

  try {
    const data = await fetch(u('/api/login'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }).then(async r => {
      if (!r.ok) { const e = await r.json().catch(() => ({ detail: 'Login failed' })); throw new Error(e.detail); }
      return r.json();
    });
    saveSession(data.token, data.role, data.username);
    S.history = [];
    chatMessages.innerHTML = '';
    showApp();
  } catch (err) {
    loginError.textContent = err.message || 'Invalid credentials. Please try again.';
    loginError.hidden = false;
    loginPass.value = '';
    loginPass.focus();
  } finally {
    loginBtn.disabled = false;
    loginBtn.querySelector('span').textContent = 'Enter the Library';
  }
});


// ════════════════════════════════════════════════════════════════════════════
//  SIGNUP
// ════════════════════════════════════════════════════════════════════════════

function showSignupMsg(text, isErr = true) {
  signupMsg.textContent = text;
  signupMsg.className   = 'form-msg ' + (isErr ? 'form-err' : 'form-ok');
  signupMsg.hidden      = false;
}

signupForm.addEventListener('submit', async e => {
  e.preventDefault();
  const username = signupUser.value.trim();
  const email    = signupEmail.value.trim();
  const pass     = signupPass.value;
  const passC    = signupPassC.value;

  // Client-side validation
  if (username.length < 3) return showSignupMsg('Username must be at least 3 characters.');
  if (!email.includes('@')) return showSignupMsg('Please enter a valid email address.');
  if (pass.length < 6)      return showSignupMsg('Password must be at least 6 characters.');
  if (pass !== passC)        return showSignupMsg('Passwords do not match.');

  signupBtn.disabled = true;
  signupBtn.querySelector('span').textContent = 'Creating…';
  signupMsg.hidden = true;

  try {
    await fetch(u('/api/signup'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password: pass }),
    }).then(async r => {
      if (!r.ok) { const e = await r.json().catch(() => ({ detail: 'Signup failed' })); throw new Error(e.detail); }
      return r.json();
    });

    showSignupMsg('✓ Account created! You can now sign in.', false);
    signupForm.reset();
    // Auto-switch to login after 1.8 s
    setTimeout(() => {
      goLogin.click();
      loginUser.value = username;
      loginPass.focus();
    }, 1800);
  } catch (err) {
    showSignupMsg(err.message || 'Sign-up failed. Please try again.');
  } finally {
    signupBtn.disabled = false;
    signupBtn.querySelector('span').textContent = 'Create Account';
  }
});


// ════════════════════════════════════════════════════════════════════════════
//  LOGOUT
// ════════════════════════════════════════════════════════════════════════════

logoutBtn.addEventListener('click', async () => {
  try { await fetch(u('/api/logout'), { method: 'POST', headers: authHeaders() }); } catch { /* ignore */ }
  clearSession();
  S.history = [];
  chatMessages.innerHTML = '';
  closeAdminPanel();
  showAuth();
});


// ════════════════════════════════════════════════════════════════════════════
//  SCREEN SWITCHING
// ════════════════════════════════════════════════════════════════════════════

function showAuth() {
  authScreen.classList.remove('hidden');
  appScreen.classList.add('hidden');
  // Reset card positions
  loginCard.className  = 'auth-card';
  signupCard.className = 'auth-card auth-card-right';
  loginUser.value = loginPass.value = '';
  loginError.hidden = true;
}

function showApp() {
  authScreen.classList.add('hidden');
  appScreen.classList.remove('hidden');

  userDisplay.textContent = S.username;
  userAvatar.textContent  = S.username[0]?.toUpperCase() || 'U';
  roleDisplay.textContent = S.role;
  roleDisplay.className   = 'role-badge' + (S.role === 'admin' ? ' role-admin' : '');

  if (S.role === 'admin') {
    adminToggleBtn.classList.remove('hidden');
    loadStats();
    loadUsers();
    loadUrlSources();
  } else {
    adminToggleBtn.classList.add('hidden');
  }

  if (!chatMessages.children.length) {
    appendMessage('assistant', `Welcome, ${S.username}! Ask me anything about Nalanda Mahavihara — its history, architecture, sculptures, conservation, or UNESCO designation.`, {});
  }
}


// ════════════════════════════════════════════════════════════════════════════
//  ADMIN PANEL TOGGLE
// ════════════════════════════════════════════════════════════════════════════

function openAdminPanel() {
  adminPanel.classList.add('open');
  adminPanel.setAttribute('aria-hidden', 'false');
  mainContent.classList.add('panel-open');
  adminToggleBtn.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M18 6L6 18M6 6l12 12"/></svg>
    Close`;
}

function closeAdminPanel() {
  adminPanel.classList.remove('open');
  adminPanel.setAttribute('aria-hidden', 'true');
  mainContent.classList.remove('panel-open');
  adminToggleBtn.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
    Manage`;
}

adminToggleBtn.addEventListener('click', () =>
  adminPanel.classList.contains('open') ? closeAdminPanel() : openAdminPanel()
);
closePanelBtn.addEventListener('click', closeAdminPanel);


// ════════════════════════════════════════════════════════════════════════════
//  PANEL TABS
// ════════════════════════════════════════════════════════════════════════════

$$('.ptab').forEach(tab => {
  tab.addEventListener('click', () => {
    $$('.ptab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const target = tab.dataset.tab;
    $('tabKnowledge').classList.toggle('hidden', target !== 'knowledge');
    $('tabUsers').classList.toggle('hidden',     target !== 'users');
    if (target === 'users') loadUsers();
  });
});


// ════════════════════════════════════════════════════════════════════════════
//  INDEX MODE RADIO
// ════════════════════════════════════════════════════════════════════════════



// ════════════════════════════════════════════════════════════════════════════
//  ADMIN: STATS
// ════════════════════════════════════════════════════════════════════════════

async function loadStats() {
  try {
    const d = await apiGet('/api/admin/stats');
    statVectors.textContent = d.summary_vectors  ?? '?';
    statDocs.textContent    = d.docstore_entries ?? '?';
  } catch {
    statVectors.textContent = statDocs.textContent = 'Err';
  }
}
refreshStatsBtn.addEventListener('click', loadStats);


// ════════════════════════════════════════════════════════════════════════════
//  ADMIN: PDF LIST
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
//  ADMIN: UPLOAD & AUTO-INDEX
// ════════════════════════════════════════════════════════════════════════════

uploadZone.addEventListener('click',   () => pdfFileInput.click());
uploadZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') pdfFileInput.click(); });
$('browseFilesLink').addEventListener('click',  e => { e.stopPropagation(); pdfFileInput.click(); });
$('browseFolderLink').addEventListener('click', e => { e.stopPropagation(); folderInput.click(); });

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  const files = [...(e.dataTransfer.files || [])].filter(f => f.name.toLowerCase().endsWith('.pdf'));
  if (files.length) uploadFiles(files);
});
pdfFileInput.addEventListener('change', () => {
  const files = [...pdfFileInput.files];
  pdfFileInput.value = '';
  if (files.length) uploadFiles(files);
});
folderInput.addEventListener('change', () => {
  const files = [...folderInput.files].filter(f => f.name.toLowerCase().endsWith('.pdf'));
  folderInput.value = '';
  if (!files.length) { showUploadStatus('No PDF files found in that folder.', 'error'); return; }
  uploadFiles(files);
});

// ── Step 1: Upload ────────────────────────────────────────────────────────────
async function uploadFiles(files) {
  uploadedPanel.classList.add('hidden');
  indexProgress.classList.add('hidden');
  progressLog.innerHTML = '';
  uploadZone.style.pointerEvents = 'none';
  showUploadStatus(`Uploading ${files.length} file(s)…`, '');

  const uploaded = [];
  let fail = 0;
  for (const file of files) {
    try {
      const form = new FormData();
      form.append('file', file);
      const r = await fetch(u('/api/admin/upload'), {
        method: 'POST', headers: { Authorization: `Bearer ${S.token}` }, body: form,
      });
      if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail); }
      uploaded.push(file.name);
    } catch { fail++; }
  }

  uploadZone.style.pointerEvents = '';
  if (!uploaded.length) { showUploadStatus(`Upload failed for all ${fail} file(s).`, 'error'); return; }

  const msg = fail ? `Uploaded ${uploaded.length} of ${files.length} — select which to index:` : `Uploaded ${uploaded.length} file(s) — select which to index:`;
  showUploadStatus(msg, fail ? '' : 'success');
  showUploadedPanel(uploaded);
}

function showUploadedPanel(fileNames) {
  uploadedList.innerHTML = fileNames.map(name => `
    <label class="uploaded-item">
      <input type="checkbox" class="file-chk" data-name="${esc(name)}" checked />
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;color:var(--saffron)" aria-hidden="true">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/>
      </svg>
      <span class="uploaded-item-name">${esc(name)}</span>
    </label>`).join('');
  selectAllChk.checked = true;
  updateSelectedCount();
  uploadedPanel.classList.remove('hidden');
}

function getCheckedFiles() {
  return [...uploadedList.querySelectorAll('.file-chk:checked')].map(c => c.dataset.name);
}

function updateSelectedCount() {
  const n = getCheckedFiles().length;
  const total = uploadedList.querySelectorAll('.file-chk').length;
  selectedCount.textContent = `${n} of ${total} selected`;
  indexSelectedBtn.textContent = n ? `→ Index Selected (${n})` : '→ Index Selected';
  indexSelectedBtn.disabled = n === 0;
}

selectAllChk.addEventListener('change', () => {
  uploadedList.querySelectorAll('.file-chk').forEach(c => { c.checked = selectAllChk.checked; });
  updateSelectedCount();
});
uploadedList.addEventListener('change', e => {
  if (e.target.classList.contains('file-chk')) {
    const all = uploadedList.querySelectorAll('.file-chk');
    selectAllChk.checked = [...all].every(c => c.checked);
    updateSelectedCount();
  }
});

// ── Step 2: Index selected ────────────────────────────────────────────────────
indexSelectedBtn.addEventListener('click', async () => {
  const toIndex = getCheckedFiles();
  if (!toIndex.length) return;
  uploadedPanel.classList.add('hidden');
  indexProgress.classList.remove('hidden');
  indexSelectedBtn.disabled = true;

  for (let i = 0; i < toIndex.length; i++) {
    const name = toIndex[i];
    addLog(`[${i + 1}/${toIndex.length}] Indexing ${name}…`);
    try {
      const job = await apiPost('/api/admin/index', { mode: 'single', pdf_path: name });
      S.jobId = job.job_id;
      await pollUntilDone();
    } catch (err) {
      addLog(`✗ ${name}: ${err.message}`, 'err');
    }
  }

  addLog(`✓ Done — ${toIndex.length} file(s) indexed.`, 'done');
  showUploadStatus(`✓ ${toIndex.length} file(s) indexed and ready.`, 'success');
  loadStats();
});

function pollUntilDone() {
  return new Promise((resolve, reject) => {
    let _last = 0;
    S.jobTimer = setInterval(async () => {
      try {
        const j = await apiGet(`/api/admin/index/${S.jobId}`);
        j.messages.slice(_last).forEach(m => addLog('  ' + m));
        _last = j.messages.length;
        if (j.status === 'done') {
          clearInterval(S.jobTimer); S.jobTimer = null; resolve();
        } else if (j.status === 'error') {
          clearInterval(S.jobTimer); S.jobTimer = null; reject(new Error(j.error));
        }
      } catch (err) {
        clearInterval(S.jobTimer); S.jobTimer = null; reject(err);
      }
    }, 1500);
  });
}

function showUploadStatus(msg, type) {
  uploadStatus.textContent = msg;
  uploadStatus.className = 'upload-status' + (type ? ` ${type}` : '');
  uploadStatus.classList.remove('hidden');
}

function addLog(msg, type = '') {
  const line = document.createElement('div');
  line.className = 'log-line' + (type ? ` log-${type}` : '');
  line.textContent = msg;
  progressLog.appendChild(line);
  progressLog.scrollTop = progressLog.scrollHeight;
}

clearIndexBtn.addEventListener('click', () => {
  showConfirm('Clear Entire Index',
    'This permanently deletes all vectors and stored documents. This cannot be undone.',
    async () => {
      try {
        await apiDelete('/api/admin/index');
        loadStats();
        indexProgress.classList.remove('hidden');
        progressLog.innerHTML = '';
        addLog('Index cleared.', 'done');
      } catch (err) { alert(`Failed: ${err.message}`); }
    });
});


// ════════════════════════════════════════════════════════════════════════════
//  ADMIN: USER MANAGEMENT
// ════════════════════════════════════════════════════════════════════════════

async function loadUsers() {
  userList.innerHTML = '<p class="muted-text">Loading…</p>';
  try {
    const { users = [] } = await apiGet('/api/admin/users');
    if (!users.length) { userList.innerHTML = '<p class="muted-text">No users found.</p>'; return; }
    userList.innerHTML = '';
    users.forEach(u => userList.appendChild(buildUserRow(u)));
  } catch (err) {
    userList.innerHTML = `<p class="muted-text">Error: ${esc(err.message)}</p>`;
  }
}
refreshUsersBtn.addEventListener('click', loadUsers);

function buildUserRow(u) {
  const node = userRowTpl.content.firstElementChild.cloneNode(true);
  node.dataset.userId = u.id;

  const av = node.querySelector('.user-row-av');
  av.textContent = u.username[0]?.toUpperCase() || 'U';

  node.querySelector('.user-row-name').textContent  = u.username;
  node.querySelector('.user-row-email').textContent = u.email;

  const badge = node.querySelector('.user-row-badge');
  if (!u.is_active) {
    badge.textContent = 'Inactive'; badge.className = 'user-row-badge inactive';
  } else {
    badge.textContent = u.role; badge.className = 'user-row-badge' + (u.role === 'admin' ? ' admin' : '');
  }

  const actions = node.querySelector('.user-row-actions');

  // Toggle active
  const toggleBtn = document.createElement('button');
  toggleBtn.className = 'user-action-btn';
  toggleBtn.textContent = u.is_active ? 'Deactivate' : 'Activate';
  toggleBtn.addEventListener('click', async () => {
    try { await apiPatch(`/api/admin/users/${u.id}`, { is_active: u.is_active ? 0 : 1 }); loadUsers(); }
    catch (err) { alert(`Failed: ${err.message}`); }
  });
  actions.appendChild(toggleBtn);

  // Toggle role
  if (u.is_active) {
    const roleBtn = document.createElement('button');
    roleBtn.className = 'user-action-btn promote';
    roleBtn.textContent = u.role === 'admin' ? '→ User' : '→ Admin';
    roleBtn.addEventListener('click', async () => {
      try { await apiPatch(`/api/admin/users/${u.id}`, { role: u.role === 'admin' ? 'user' : 'admin' }); loadUsers(); }
      catch (err) { alert(`Failed: ${err.message}`); }
    });
    actions.appendChild(roleBtn);
  }

  // Delete
  const delBtn = document.createElement('button');
  delBtn.className = 'user-action-btn del';
  delBtn.textContent = 'Delete';
  delBtn.addEventListener('click', () => {
    showConfirm('Delete User', `Permanently delete "${u.username}"? This cannot be undone.`, async () => {
      try { await apiDelete(`/api/admin/users/${u.id}`); loadUsers(); }
      catch (err) { alert(`Failed: ${err.message}`); }
    });
  });
  actions.appendChild(delBtn);

  return node;
}


// ════════════════════════════════════════════════════════════════════════════
//  CONFIRM DIALOG
// ════════════════════════════════════════════════════════════════════════════

let _confirmCb = null;
function showConfirm(title, msg, onOk) {
  confirmTitle.textContent = title;
  confirmMsg.textContent   = msg;
  _confirmCb = onOk;
  confirmOverlay.classList.remove('hidden');
}
confirmOk.addEventListener('click', async () => {
  confirmOverlay.classList.add('hidden');
  if (_confirmCb) { await _confirmCb(); _confirmCb = null; }
});
confirmCancel.addEventListener('click', () => { confirmOverlay.classList.add('hidden'); _confirmCb = null; });
confirmOverlay.addEventListener('click', e => {
  if (e.target === confirmOverlay) { confirmOverlay.classList.add('hidden'); _confirmCb = null; }
});


// ════════════════════════════════════════════════════════════════════════════
//  ADMIN: WEB URL SOURCES
// ════════════════════════════════════════════════════════════════════════════

async function loadUrlSources() {
  if (!urlSourceList) return;
  try {
    const { url_sources = [] } = await apiGet('/api/admin/url-sources');
    urlSourceList.innerHTML = '';
    if (!url_sources.length) {
      urlSourceList.innerHTML = '<p class="muted-text" style="font-size:.76rem">No URLs configured yet.</p>';
      return;
    }
    url_sources.forEach(src => {
      let hostname;
      try { hostname = new URL(src.url).hostname; } catch { hostname = src.url; }
      const chip = document.createElement('div');
      chip.className = 'url-chip';
      chip.innerHTML =
        `<span class="url-chip-text" title="${esc(src.url)}">${esc(hostname)}</span>` +
        `<button type="button" class="url-chip-remove" data-id="${src.id}" aria-label="Remove">×</button>`;
      urlSourceList.appendChild(chip);
    });
  } catch (err) {
    if (urlSourceList) urlSourceList.innerHTML = `<p class="muted-text">Error: ${esc(err.message)}</p>`;
  }
}

if (urlSourceList) {
  urlSourceList.addEventListener('click', e => {
    const btn = e.target.closest('.url-chip-remove');
    if (!btn) return;
    const chip = btn.closest('.url-chip');
    const urlText = chip?.querySelector('.url-chip-text')?.title || 'this URL';
    showConfirm('Remove URL Source',
      `Remove "${urlText}" from the knowledge base? Its indexed content will be deleted.`,
      async () => {
        try {
          await apiDelete(`/api/admin/url-sources/${btn.dataset.id}`);
          loadUrlSources();
        } catch (err) { alert(`Failed: ${err.message}`); }
      });
  });
}

async function addUrlSource() {
  if (!urlSourceInput) return;
  let url = urlSourceInput.value.trim();
  if (!url) return;
  if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
  try { new URL(url); } catch { alert('Invalid URL'); return; }
  urlSourceStatus.textContent = 'Checking URL…';
  urlSourceStatus.className   = 'upload-status';
  urlSourceStatus.classList.remove('hidden');
  try {
    await apiPost('/api/admin/url-sources', { url });
    urlSourceInput.value = '';
    urlSourceStatus.textContent = '✓ URL saved. Content will be fetched and sent to the AI on next user query.';
    urlSourceStatus.className   = 'upload-status success';
    setTimeout(() => urlSourceStatus.classList.add('hidden'), 4000);
    loadUrlSources();
  } catch (err) {
    urlSourceStatus.textContent = `Error: ${err.message}`;
    urlSourceStatus.className   = 'upload-status error';
  }
}

if (urlSourceAddBtn) urlSourceAddBtn.addEventListener('click', addUrlSource);
if (urlSourceInput)  urlSourceInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); addUrlSource(); }
});


// ════════════════════════════════════════════════════════════════════════════
//  CHAT
// ════════════════════════════════════════════════════════════════════════════

chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
  const len = chatInput.value.length;
  charCount.textContent = `${len} / 1000`;
  charCount.classList.toggle('warn', len > 900);
  sendBtn.disabled = chatInput.value.trim().length === 0;
});

chatInput.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!sendBtn.disabled) chatForm.dispatchEvent(new Event('submit'));
  }
});

chatForm.addEventListener('submit', async e => {
  e.preventDefault();
  const q = chatInput.value.trim();
  if (!q) return;

  suggestions.style.display = 'none';
  chatInput.value = ''; chatInput.style.height = 'auto';
  sendBtn.disabled = true; charCount.textContent = '0 / 1000';

  appendMessage('user', q, {});
  const historySnapshot = S.history.slice(-8); // capture history BEFORE adding current turn
  S.history.push({ role: 'user', content: q });
  thinkingRow.classList.remove('hidden');
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const response = await fetch(u('/api/chat/stream'), {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({
        question: q,
        top_k: parseInt(topKInput.value || '4', 10),
        history: historySnapshot,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    thinkingRow.classList.add('hidden');
    const { node, textEl, imagesEl, metaEl } = createAssistantBubble();

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullAnswer = '';
    let donePayload = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch { continue; }
        if (event.type === 'heartbeat') {
          continue;
        } else if (event.type === 'token') {
          fullAnswer += event.content;
          textEl.textContent = fullAnswer;
          chatMessages.scrollTop = chatMessages.scrollHeight;
        } else if (event.type === 'done') {
          donePayload = event;
          if (event.answer) { fullAnswer = event.answer; textEl.textContent = fullAnswer; }
        } else if (event.type === 'error') {
          textEl.textContent = `Query failed: ${event.message}`;
        }
      }
    }

    if (donePayload) {
      const images = donePayload.images || [];
      const captions = donePayload.captions || [];
      images.forEach((b64, i) => {
        const wrap = document.createElement('div');
        wrap.className = 'img-wrap';
        const img = document.createElement('img');
        img.loading = 'lazy'; img.alt = captions[i] || `Image ${i + 1}`;
        img.src = `data:image/jpeg;base64,${b64}`;
        wrap.appendChild(img);
        if (captions[i]) {
          const cap = document.createElement('p');
          cap.className = 'img-caption'; cap.textContent = captions[i];
          wrap.appendChild(cap);
        }
        imagesEl.appendChild(wrap);
      });

      if (donePayload.web_searched) {
        metaEl.innerHTML = '<span class="source-badge web">🌐 Web search</span>';
        verifiedBadge.classList.remove('hidden', 'ok', 'no');
        verifiedBadge.textContent = '🌐 Web';
        verifiedBadge.classList.add('ok');
        verifiedBadge.classList.remove('hidden');
      } else {
        metaEl.textContent = '';
        verifiedBadge.classList.add('hidden');
      }

      S.history.push({ role: 'assistant', content: fullAnswer });
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  } catch (err) {
    thinkingRow.classList.add('hidden');
    if (err.message.includes('401') || err.message.toLowerCase().includes('session')) {
      appendMessage('assistant', 'Your session has expired. Please log in again.', {});
      clearSession(); setTimeout(showAuth, 1800);
    } else {
      appendMessage('assistant', `Query failed: ${err.message}`, {});
    }
  } finally {
    sendBtn.disabled = false;
  }
});

function createAssistantBubble() {
  const node = msgTpl.content.firstElementChild.cloneNode(true);
  node.classList.add('assistant');
  node.querySelector('.msg-avatar').textContent = 'N';
  const textEl   = node.querySelector('.msg-text');
  const imagesEl = node.querySelector('.msg-images');
  const metaEl   = node.querySelector('.msg-meta');
  textEl.textContent = '';
  chatMessages.appendChild(node);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return { node, textEl, imagesEl, metaEl };
}

function appendMessage(role, text, payload) {
  const node = msgTpl.content.firstElementChild.cloneNode(true);
  node.classList.add(role);

  const av = node.querySelector('.msg-avatar');
  av.textContent = role === 'user' ? (S.username[0] || 'U').toUpperCase() : 'N';

  node.querySelector('.msg-text').textContent = text || '';

  const imagesEl  = node.querySelector('.msg-images');
  const images    = payload.images   || [];
  const captions  = payload.captions || [];
  images.forEach((b64, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'img-wrap';
    const img = document.createElement('img');
    img.loading = 'lazy'; img.alt = captions[i] || `Image ${i + 1}`;
    img.src = `data:image/jpeg;base64,${b64}`;
    wrap.appendChild(img);
    if (captions[i]) {
      const cap = document.createElement('p');
      cap.className = 'img-caption'; cap.textContent = captions[i];
      wrap.appendChild(cap);
    }
    imagesEl.appendChild(wrap);
  });

  const metaEl = node.querySelector('.msg-meta');
  if (role === 'assistant' && payload.web_searched) {
    metaEl.innerHTML = '<span class="source-badge web">🌐 Web search</span>';
  }

  chatMessages.appendChild(node);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

clearChatBtn.addEventListener('click', () => {
  chatMessages.innerHTML = ''; S.history = [];
  verifiedBadge.classList.add('hidden');
  suggestions.style.display = '';
  appendMessage('assistant', 'Chat cleared. Ask me anything about Nalanda Mahavihara!', {});
});

$$('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chatInput.value = chip.textContent.trim();
    chatInput.dispatchEvent(new Event('input'));
    chatInput.focus();
  });
});


// ════════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ════════════════════════════════════════════════════════════════════════════

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtSize(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b/1024).toFixed(0)} KB`;
  return `${(b/1048576).toFixed(1)} MB`;
}


// ════════════════════════════════════════════════════════════════════════════
//  INIT
// ════════════════════════════════════════════════════════════════════════════

function init() {
  initParallax();
  if (S.token) {
    fetch(u('/api/health'), { headers: { Authorization: `Bearer ${S.token}` } })
      .then(r => r.ok ? showApp() : (clearSession(), showAuth()))
      .catch(() => { clearSession(); showAuth(); });
  } else {
    showAuth();
  }
}

init();
