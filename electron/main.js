// Electron main process.
// Responsibilities: open the app window, manage app-data folders, spawn/monitor
// the Python inference sidecar, and bridge renderer <-> sidecar over IPC.
// Pattern mirrors the pdfToAudio project's electron/main.js.
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const isDev = !app.isPackaged;
const VITE_DEV_SERVER_URL = 'http://localhost:5173';

let mainWindow = null;
let sidecarProcess = null;
let sidecarPort = 8843; // fixed local-only port for the FastAPI sidecar (distinct from pdfToAudio's 8756 so both can run in dev)

// ---- App-data layout ----
// models:   cached base-model weights (Flux) and style-extraction adapter weights
// essences: saved Essence folders (embedding + metadata) — see sidecar/paths.py
// media:    every finished creation (Media Page archive)
// cache:    scratch/intermediate files
// db:       local index (essence/media metadata) if/when a DB is added
function appDataDirs() {
  const root = path.join(app.getPath('userData'));
  const dirs = {
    root,
    models: path.join(root, 'models'),
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
// Dev: run the venv-installed `python` against sidecar/app.py.
// Packaged: run the PyInstaller-built binary from process.resourcesPath/sidecar.
// (No packaged build exists yet — that branch mirrors pdfToAudio's approach
// and will be wired up once there's a pipeline worth shipping.)
function startSidecar() {
  const dirs = appDataDirs();

  let cmd;
  let args;
  if (isDev) {
    cmd = process.platform === 'win32'
      ? path.join(__dirname, '..', 'sidecar', 'venv', 'Scripts', 'python.exe')
      : path.join(__dirname, '..', 'sidecar', 'venv', 'bin', 'python');
    args = [path.join(__dirname, '..', 'sidecar', 'app.py')];
  } else {
    const exeName = process.platform === 'win32' ? 'sidecar.exe' : 'sidecar';
    cmd = path.join(process.resourcesPath, 'sidecar', exeName);
    args = [];
  }

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
      const res = await fetch(`http://127.0.0.1:${sidecarPort}/health`);
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

// Generic proxy to the local-only sidecar HTTP API — kept in the main process
// (not called directly from the renderer) so the renderer never needs its own
// network access, matching the sandboxed/no-nodeIntegration webPreferences.
ipcMain.handle('sidecar:call', async (_event, { method = 'GET', path, body } = {}) => {
  const url = `http://127.0.0.1:${sidecarPort}${path}`;
  const opts = { method };
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

// Reads a local image file and returns it as a data: URL. The renderer is
// served from http://localhost:5173 in dev (a different origin from
// file://), and Chromium's cross-origin restrictions block <img src="file://...">
// from an http(s) page — so local images picked via the native dialog or
// drag-and-drop are routed through here rather than as file:// URLs.
ipcMain.handle('image:readAsDataUrl', async (_event, filePath) => {
  const buf = await fs.promises.readFile(filePath);
  const ext = path.extname(filePath).slice(1).toLowerCase();
  const mime = ext === 'jpg' ? 'jpeg' : ext || 'png';
  return `data:image/${mime};base64,${buf.toString('base64')}`;
});

ipcMain.handle('dialog:openImage', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose an image',
    properties: ['openFile'],
    filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg', 'webp'] }],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle('shell:showInFolder', (_event, filePath) => {
  shell.showItemInFolder(filePath);
});

app.whenReady().then(() => {
  appDataDirs();
  startSidecar();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (sidecarProcess) {
    sidecarProcess.kill();
    sidecarProcess = null;
  }
});
