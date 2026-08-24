import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: './' so built asset paths work when loaded via file:// from
// Electron's packaged app (loadFile), not just from an http server.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
  },
});
