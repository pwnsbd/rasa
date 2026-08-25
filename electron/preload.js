// Preload script: the only bridge between the sandboxed renderer and Node/Electron.
// Keep this surface small and explicit — renderer never gets raw fs/child_process access.
const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('appBridge', {
  getSidecarHealth: () => ipcRenderer.invoke('sidecar:health'),
  getAppDirs: () => ipcRenderer.invoke('app:dirs'),

  // Generic sidecar HTTP proxy — see electron/main.js for why this goes
  // through the main process rather than the renderer calling fetch directly.
  sidecarCall: (method, path, body) => ipcRenderer.invoke('sidecar:call', { method, path, body }),
  getSidecarBaseUrl: () => ipcRenderer.invoke('sidecar:baseUrl'),

  openImageDialog: () => ipcRenderer.invoke('dialog:openImage'),
  // Multi-select variant for the Cauldron (blend/distill from several
  // photos at once) — everything else still uses the single-select dialog.
  openImagesDialog: () => ipcRenderer.invoke('dialog:openImages'),
  readImageAsDataUrl: (filePath) => ipcRenderer.invoke('image:readAsDataUrl', filePath),
  // Resolves a dropped File object (from a drag-and-drop event) to its
  // absolute filesystem path — File.path is unavailable on sandboxed
  // renderer File objects; webUtils.getPathForFile is the replacement.
  getPathForFile: (file) => webUtils.getPathForFile(file),

  showInFolder: (filePath) => ipcRenderer.invoke('shell:showInFolder', filePath),

  // First-run sidecar setup progress (packaged installs only — see
  // electron/sidecarBootstrap.js; dev always uses the pre-built
  // sidecar/venv and never fires this). getCurrentBootstrapStatus covers
  // the race where the renderer mounts after an earlier push already went
  // out — main.js remembers the latest and hands it back on request.
  onBootstrapProgress: (callback) => {
    const listener = (_event, status) => callback(status);
    ipcRenderer.on('bootstrap:progress', listener);
    return () => ipcRenderer.removeListener('bootstrap:progress', listener);
  },
  getCurrentBootstrapStatus: () => ipcRenderer.invoke('bootstrap:currentStatus'),
});
