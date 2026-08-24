// Dev QA helper: launch the Electron app via Playwright, wait for the
// current screen to settle, screenshot it, quit. Adapted from pdfToAudio's
// scripts/verify-launch.mjs (with its duplicate `const deadline` fixed).
// Requires the Vite dev server already running (npm run dev:ui) and
// playwright-core installed (npm install --no-save playwright-core).
// Usage: node scripts/verify-launch.mjs [output-dir] [shot-name]
import { _electron as electron } from 'playwright-core';
import path from 'node:path';
import fs from 'node:fs';

const APP_DIR = path.resolve(import.meta.dirname, '..');
const SHOT_DIR = process.argv[2] || path.join(APP_DIR, '.tmp-shots');
const SHOT_NAME = process.argv[3] || 'screen';
fs.mkdirSync(SHOT_DIR, { recursive: true });

const electronBin = process.platform === 'darwin'
  ? path.join(APP_DIR, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(APP_DIR, 'node_modules/electron/dist/electron.exe');

async function main() {
  console.log('launching electron from', APP_DIR);
  const app = await electron.launch({
    executablePath: electronBin,
    args: [APP_DIR],
    timeout: 30_000,
  });

  console.log('launched. windows so far:', app.windows().length);

  // firstWindow() races against Electron's auto-opened DevTools window in
  // dev mode and can resolve to devtools://... instead of the real app page.
  // Poll for a real non-devtools window instead.
  let page = null;
  let deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    page = app.windows().find((w) => !w.url().startsWith('devtools://') && w.url() !== 'about:blank');
    if (page) break;
    await new Promise((r) => setTimeout(r, 300));
  }
  if (!page) throw new Error('No non-devtools window appeared within 15s');
  console.log('app window url:', page.url());

  // Poll for a settled state instead of a blind sleep — the Settings/Main
  // Stage screens start with a "Loading…"/"Checking sidecar…" placeholder
  // until their first sidecar call resolves.
  let text = '';
  deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    text = await page.evaluate(() => document.body.innerText).catch(() => '');
    if (text && !/Checking sidecar|Loading/.test(text)) break;
    await new Promise((r) => setTimeout(r, 500));
  }

  console.log('--- rendered text ---');
  console.log(text);
  console.log('---------------------');

  const shotPath = path.join(SHOT_DIR, `${SHOT_NAME}.png`);
  await page.screenshot({ path: shotPath });
  console.log('screenshot saved to', shotPath);

  await app.close();
}

main().catch((err) => {
  console.error('FAILED:', err);
  process.exit(1);
});
