// One-off smoke test (not part of the regular QA scripts): launches the
// REAL packaged exe (release/win-unpacked/Rasa.exe, isPackaged=true) to
// confirm the first-run bootstrap overlay actually renders — as opposed to
// electron/sidecarBootstrap.js's underlying logic, already verified
// directly via Node. Screenshots fast and closes immediately, well before
// the real (multi-GB) pip install would get underway, since this is only
// checking that the UI shows up correctly, not running a full install.
import { _electron as electron } from 'playwright-core';
import path from 'node:path';
import fs from 'node:fs';

const APP_DIR = path.resolve(import.meta.dirname, '..');
const EXE = path.join(APP_DIR, 'release', 'win-unpacked', 'Rasa.exe');
const SHOT_DIR = path.join(APP_DIR, '.tmp-shots');
fs.mkdirSync(SHOT_DIR, { recursive: true });

async function main() {
  console.log('launching packaged exe:', EXE);
  const app = await electron.launch({ executablePath: EXE, timeout: 30_000 });

  let page = null;
  let deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    page = app.windows().find((w) => !w.url().startsWith('devtools://') && w.url() !== 'about:blank');
    if (page) break;
    await new Promise((r) => setTimeout(r, 200));
  }
  if (!page) throw new Error('No window appeared within 15s');
  console.log('window url:', page.url());

  // Give the bootstrap a few seconds to start and the overlay to mount —
  // NOT waiting for it to finish (that's a multi-GB install), just enough
  // to confirm the UI itself renders correctly.
  await new Promise((r) => setTimeout(r, 4000));

  const text = await page.evaluate(() => document.body.innerText).catch(() => '(could not read body text)');
  console.log('--- rendered text ---');
  console.log(text);
  console.log('---------------------');

  const shotPath = path.join(SHOT_DIR, 'packaged-first-run.png');
  await page.screenshot({ path: shotPath });
  console.log('screenshot saved to', shotPath);

  await app.close();
}

main().catch((err) => {
  console.error('FAILED:', err);
  process.exit(1);
});
