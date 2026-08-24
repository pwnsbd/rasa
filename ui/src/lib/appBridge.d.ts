export interface GpuInfo {
  torch_installed: boolean;
  cuda_available: boolean;
  device_name: string | null;
  device_count: number;
  compute_capability: string | null;
  vram_gb: number | null;
  cuda_runtime_version: string | null;
  torch_version: string | null;
  warning: string | null;
}

export interface SidecarHealth {
  ok: boolean;
  gpu: GpuInfo;
  dirs: Record<string, string>;
  error?: string;
}

export interface SidecarCallResult<T = unknown> {
  ok: boolean;
  status?: number;
  data?: T;
  error?: string;
}

export interface AppBridge {
  getSidecarHealth: () => Promise<SidecarHealth>;
  getAppDirs: () => Promise<Record<string, string>>;
  sidecarCall: <T = unknown>(method: string, path: string, body?: unknown) => Promise<SidecarCallResult<T>>;
  getSidecarBaseUrl: () => Promise<string>;
  openImageDialog: () => Promise<string | null>;
  getPathForFile: (file: File) => string;
  showInFolder: (filePath: string) => Promise<void>;
}

declare global {
  interface Window {
    appBridge: AppBridge;
  }
}
