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
  // Resolves a dropped File object (from a drag-and-drop event) to its
  // absolute filesystem path — File.path is unavailable on sandboxed
  // renderer File objects; webUtils.getPathForFile is the replacement.
  getPathForFile: (file) => webUtils.getPathForFile(file),

  showInFolder: (filePath) => ipcRenderer.invoke('shell:showInFolder', filePath),
});
