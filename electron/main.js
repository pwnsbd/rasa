// Electron main process.
// Responsibilities: open the app window, manage app-data folders, spawn/monitor
// the Python inference sidecar, and bridge renderer <-> sidecar over IPC.
// Pattern mirrors the pdfToAudio project's electron/main.js.
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { ensureSidecarRuntime, cancelSetup } = require('./sidecarBootstrap');

const isDev = !app.isPackaged;
const primaryInstance = app.requestSingleInstanceLock();
if (!primaryInstance) app.quit();
app.on('second-instance', () => {
  if (mainWindow) { if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.focus(); }
});
const VITE_DEV_SERVER_URL = 'http://localhost:5173';

let mainWindow = null;
let sidecarProcess = null;
let sidecarPort = 8843; // fixed local-only port for the FastAPI sidecar (distinct from pdfToAudio's 8756 so both can run in dev)

// Dev-only: keep all app data — including Electron's own internal caches
// (GPUCache, Code Cache, Session Storage, etc.) — on whichever drive the
// project itself lives on, not wherever Electron's OS-default userData
// happens to be (usually %APPDATA%, the C: drive on Windows). A ~11.5GB
// model cache landing on a drive the user doesn't want it on isn't a
// preference to leave to an env var someone has to remember every launch —
// it's the default, for `npm run dev`. Must run before any
// app.getPath('userData') call, hence right here at module top rather than
// inside appDataDirs(). RASA_MODELS_DIR (see appDataDirs() below) still
// exists as a further override for anyone who wants the model cache
// specifically somewhere else again.
//
// A PACKAGED install must NOT do this: __dirname then resolves inside the
// installed app's own resources folder (e.g. under Program Files on a
// per-machine install), and writing an ~11.5GB model cache there means
// every model download needs admin rights, and app data living next to the
// binary is unusual and easy to lose on update/uninstall. Packaged builds
// use Electron's normal per-user OS-default userData location instead —
// exactly what it's for — with RASA_MODELS_DIR still available for anyone
// who wants the model cache elsewhere.
if (isDev) {
  app.setPath('userData', path.join(__dirname, '..', 'appdata'));
} else {
  // Packaged: keep ALL app data (models, runtime, essences, media) in a
  // `data` folder inside the install directory, so removing the install
  // folder removes everything. The installer's custom macros (build/installer.nsh)
  // preserve it across upgrades. If the install dir isn't writable (e.g. the
  // user picked Program Files), fall back to Electron's default userData.
  const installData = path.join(path.dirname(process.execPath), 'data');
  try {
    fs.mkdirSync(installData, { recursive: true });
    fs.accessSync(installData, fs.constants.W_OK);
    app.setPath('userData', installData);
  } catch (_) { /* keep default userData */ }
}

// ---- App-data layout ----
// models:   cached base-model weights (SDXL/InstantStyle/ControlNet) — ~11.5GB,
//           by far the largest of these.
// essences: saved Essence folders (embedding + metadata) — see sidecar/paths.py
// media:    every finished creation (Media Page archive)
// cache:    scratch/intermediate files
// db:       local index (essence/media metadata) if/when a DB is added
function appDataDirs() {
  const root = path.join(app.getPath('userData'));
  const dirs = {
    root,
    models: process.env.RASA_MODELS_DIR || path.join(root, 'models'),
    essences: path.join(root, 'essences'),
    media: path.join(root, 'media'),
    cache: path.join(root, 'cache'),
    db: path.join(root, 'db'),
  };
  for (const dir of Object.values(dirs)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dirs;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: '#2b2233', // deep plum — avoids a white flash before the app's CSS paints
    icon: path.join(__dirname, 'icon.png'), // taskbar/window icon while running — Windows installer icon is package.json's build.win.icon (icon.ico)
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  if (isDev) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'ui', 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ---- Python sidecar lifecycle ----
// Dev: run the venv-installed `python` against sidecar/app.py — unchanged,
// this venv is built ahead of time by `npm run sidecar:setup`.
// Packaged: no system Python assumed. First launch builds a private,
// per-user Python runtime from the embeddable interpreter bundled into the
// installer (see sidecarBootstrap.js + scripts/fetch-embed-python.js);
// later launches skip straight through once that runtime already exists.
let lastBootstrapStatus = null;
let startupTask = null;
let quitting = false;
function reportStartup(status) {
  lastBootstrapStatus = status;
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('bootstrap:progress', status);
}
function launchBackend() {
  if (startupTask || sidecarProcess) return startupTask;
  startupTask = (async () => {
    try {
      reportStartup({ step: 'Preparing Rasa…' });
      await startSidecar();
      if (!quitting) await waitForSidecarHealth(120000);
    } catch (err) {
      if (sidecarProcess) { sidecarProcess.kill(); sidecarProcess = null; }
      if (!quitting) reportStartup({ step: 'Rasa could not start', detail: String(err.message || err), error: true });
    } finally { startupTask = null; }
  })();
  return startupTask;
}
ipcMain.handle('bootstrap:retry', () => { void launchBackend(); });

async function startSidecar() {
  const dirs = appDataDirs();

  let cmd;
  let args;
  if (isDev) {
    cmd = process.platform === 'win32'
      ? path.join(__dirname, '..', 'sidecar', 'venv', 'Scripts', 'python.exe')
      : path.join(__dirname, '..', 'sidecar', 'venv', 'bin', 'python');
    args = [path.join(__dirname, '..', 'sidecar', 'app.py')];
  } else {
    cmd = await ensureSidecarRuntime({
      userDataRoot: dirs.root,
      resourcesPath: process.resourcesPath,
      onProgress: (status) => {
        reportStartup(status);
      },
    });
    args = [path.join(process.resourcesPath, 'sidecar', 'app.py')];
  }

  if (quitting) return;
  // Embeddable Python's isolated path does not include the script directory.
  args = ['-c', "import runpy,sys; from pathlib import Path; p=sys.argv[1]; sys.path.insert(0,str(Path(p).parent)); runpy.run_path(p,run_name='__main__')", args[0]];
  sidecarProcess = spawn(cmd, args, {
    windowsHide: true,
    env: {
      ...process.env,
      SIDECAR_PORT: String(sidecarPort),
      APP_MODELS_DIR: dirs.models,
      APP_ESSENCES_DIR: dirs.essences,
      APP_MEDIA_DIR: dirs.media,
      APP_CACHE_DIR: dirs.cache,
      APP_DB_DIR: dirs.db,
    },
  });

  sidecarProcess.stdout.on('data', (d) => console.log(`[sidecar] ${d}`));
  sidecarProcess.stderr.on('data', (d) => console.error(`[sidecar:err] ${d}`));
  sidecarProcess.on('exit', (code) => {
    console.log(`[sidecar] exited with code ${code}`);
    if (!quitting) reportStartup({ step: 'Rasa’s processing engine stopped', detail: 'Restart Rasa to reconnect. Your saved work is kept.', error: true });
    sidecarProcess = null;
  });
  sidecarProcess.on('error', (err) => {
    console.error(`[sidecar] failed to start: ${err.stack || err}`);
  });
}

async function waitForSidecarHealth(timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`http://127.0.0.1:${sidecarPort}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return await res.json();
    } catch {
      // not up yet, keep polling
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('Sidecar did not become healthy in time');
}

ipcMain.handle('sidecar:health', async () => {
  try {
    return await waitForSidecarHealth(5000);
  } catch (err) {
    return { ok: false, error: String(err) };
  }
});

ipcMain.handle('app:dirs', () => appDataDirs());

// The renderer mounts asynchronously and can miss earlier 'bootstrap:progress'
// pushes (main.js starts the bootstrap before the window has finished loading
// — see app.whenReady() below) — this lets it catch up to whatever the
// latest status was instead of only ever seeing pushes that happen to land
// after it's ready to listen.
ipcMain.handle('bootstrap:currentStatus', () => lastBootstrapStatus);

// Generic proxy to the local-only sidecar HTTP API — kept in the main process
// (not called directly from the renderer) so the renderer never needs its own
// network access, matching the sandboxed/no-nodeIntegration webPreferences.
// No timeout was set here at all, which meant every call fell back to
// Node's built-in fetch (undici) default — not a value anyone chose for
// this app, and short enough that a real /apply (first-run model download
// on top of generation, or just a slow depth/masking pass — see recent
// perf discussion) can trip it with a raw "fetch failed / HeadersTimeoutError"
// that looks like a crash even when the sidecar is still working correctly.
// This is a 127.0.0.1 call to a process this same app spawned — there's no
// real network flakiness to guard against, so the ceiling here should come
// from the sidecar's own already-reasoned timeouts instead: pipeline_manager.
// get_pipeline_blocking's 1800s (first-run model download) plus real
// generation time on top, with headroom.
const SIDECAR_CALL_TIMEOUT_MS = 40 * 60 * 1000;

ipcMain.handle('sidecar:call', async (_event, { method = 'GET', path, body } = {}) => {
  const url = `http://127.0.0.1:${sidecarPort}${path}`;
  const opts = { method, signal: AbortSignal.timeout(SIDECAR_CALL_TIMEOUT_MS) };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const detail = (data && data.detail) || `Request failed (${res.status})`;
    return { ok: false, status: res.status, error: detail };
  }
  return { ok: true, status: res.status, data };
});

ipcMain.handle('sidecar:baseUrl', () => `http://127.0.0.1:${sidecarPort}`);

// Formats Chromium's <img> can decode directly from a data: URL. Anything
// else (HEIC/HEIF — no browser codec at all; TIFF — Chromium won't display
// it even though it'll happily <img> a raw file:// one on some platforms)
// needs to go through the sidecar's Pillow-based decoder instead, since
// Electron's main process has no image codec of its own.
const BROWSER_DISPLAYABLE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif']);

// Reads a local image file and returns it as a data: URL. The renderer is
// served from http://localhost:5173 in dev (a different origin from
// file://), and Chromium's cross-origin restrictions block <img src="file://...">
// from an http(s) page — so local images picked via the native dialog or
// drag-and-drop are routed through here rather than as file:// URLs.
ipcMain.handle('image:readAsDataUrl', async (_event, filePath) => {
  const ext = path.extname(filePath).slice(1).toLowerCase();

  if (!BROWSER_DISPLAYABLE_EXTENSIONS.has(ext)) {
    // HEIC/HEIF/TIFF/etc — ask the sidecar (pillow-heif-registered Pillow)
    // to decode and re-encode as a browser-displayable PNG preview.
    const res = await fetch(`http://127.0.0.1:${sidecarPort}/utils/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: filePath }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Sidecar could not preview ${ext} image (${res.status})`);
    }
    const { data_url } = await res.json();
    return data_url;
  }

  const buf = await fs.promises.readFile(filePath);
  const mime = ext === 'jpg' ? 'jpeg' : ext;
  return `data:image/${mime};base64,${buf.toString('base64')}`;
});

const IMAGE_DIALOG_FILTERS = [
  { name: 'Image', extensions: ['png', 'jpg', 'jpeg', 'webp', 'heic', 'heif', 'bmp', 'gif', 'tif', 'tiff'] },
];

ipcMain.handle('dialog:openImage', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose an image',
    properties: ['openFile'],
    filters: IMAGE_DIALOG_FILTERS,
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

// Multi-select variant for the Cauldron (blend/distill from several photos
// at once) — the single-select dialog above stays untouched for every
// other picker (Main Stage's target photo, Distillation Room's Single mode).
ipcMain.handle('dialog:openImages', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose reference photos',
    properties: ['openFile', 'multiSelections'],
    filters: IMAGE_DIALOG_FILTERS,
  });
  if (result.canceled) return [];
  return result.filePaths;
});

// Save-as for a converted result (a data: URL from /apply). Returns the saved
// path, or null if the user cancelled.
ipcMain.handle('image:saveDataUrl', async (_event, dataUrl, suggestedName) => {
  const m = /^data:image\/(\w+);base64,(.+)$/.exec(dataUrl || '');
  if (!m) throw new Error('Not an image data URL');
  const ext = m[1] === 'jpeg' ? 'jpg' : m[1];
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save converted image',
    defaultPath: `${suggestedName || 'rasa-result'}.${ext}`,
    filters: [{ name: 'Image', extensions: [ext] }],
  });
  if (result.canceled || !result.filePath) return null;
  await fs.promises.writeFile(result.filePath, Buffer.from(m[2], 'base64'));
  return result.filePath;
});

ipcMain.handle('shell:showInFolder', (_event, filePath) => {
  shell.showItemInFolder(filePath);
});

app.whenReady().then(async () => {
  if (!primaryInstance) return;
  appDataDirs();
  createWindow(); // first, so a packaged install's first-run bootstrap has a window to report progress into
  await launchBackend();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  quitting = true;
  cancelSetup();
  if (sidecarProcess) {
    sidecarProcess.kill();
    sidecarProcess = null;
  }
});
