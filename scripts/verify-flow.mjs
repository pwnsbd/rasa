// End-to-end smoke test: drives the full alchemy flow through Playwright —
// Distillation Room (extract via __testHooks bypass, since native dialogs
// can't be automated) -> seal -> Main Stage (essence now on shelf, choose a
// target photo, apply the essence) -> Media Page (creation was auto-saved).
// Screenshots each step.
//
// Requires: Vite dev server running (npm run dev:ui), playwright-core
// installed, sidecar venv set up. Run via (this shell sets
// ELECTRON_RUN_AS_NODE=1, which breaks Electron launch):
//   env -u ELECTRON_RUN_AS_NODE node scripts/verify-flow.mjs <ref.png> <target.png>
import { _electron as electron } from 'playwright-core';
import path from 'node:path';
import fs from 'node:fs';

const APP_DIR = path.resolve(import.meta.dirname, '..');
const SHOT_DIR = path.join(APP_DIR, '.tmp-shots');
fs.mkdirSync(SHOT_DIR, { recursive: true });

const [refPath, targetPath] = process.argv.slice(2);
if (!refPath || !targetPath) {
  console.error('Usage: node scripts/verify-flow.mjs <ref-image-path> <target-image-path>');
  process.exit(1);
}

const electronBin = path.join(APP_DIR, 'node_modules/electron/dist/electron.exe');

let shotN = 0;
async function shot(page, name) {
  const f = path.join(SHOT_DIR, `${String(++shotN).padStart(2, '0')}-${name}.png`);
  await page.screenshot({ path: f });
  console.log('screenshot:', f);
}

async function clickText(page, text) {
  const result = await page.evaluate((t) => {
    const els = [...document.querySelectorAll('button, a, [role="button"]')];
    const el = els.find((e) => e.textContent?.trim() === t) ?? els.find((e) => e.textContent?.includes(t));
    if (!el) return 'NOT_FOUND';
    el.click();
    return 'OK';
  }, text);
  console.log(`click "${text}" ->`, result);
  if (result === 'NOT_FOUND') throw new Error(`Could not find clickable element with text: ${text}`);
}

async function waitFor(page, predicate, label, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await page.evaluate(predicate).catch(() => false)) return;
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`Timed out waiting for: ${label}`);
}

async function main() {
  console.log('launching electron from', APP_DIR);
  const app = await electron.launch({ executablePath: electronBin, args: [APP_DIR], timeout: 30_000 });

  let page = null;
  let deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    page = app.windows().find((w) => !w.url().startsWith('devtools://') && w.url() !== 'about:blank');
    if (page) break;
    await new Promise((r) => setTimeout(r, 300));
  }
  if (!page) throw new Error('No app window appeared within 15s');
  console.log('app window url:', page.url());

  // Nav icons have no text — click by title attribute via a small helper.
  async function clickNav(title) {
    const r = await page.evaluate((t) => {
      const el = document.querySelector(`button[title="${t}"]`);
      if (!el) return 'NOT_FOUND';
      el.click();
      return 'OK';
    }, title);
    console.log(`click nav "${title}" ->`, r);
    if (r === 'NOT_FOUND') throw new Error(`nav button not found: ${title}`);
  }

  await shot(page, 'main-stage-empty');

  await clickNav('Distillation Room');
  await shot(page, 'distillation-idle');

  await page.evaluate((p) => window.__testHooks.distillation.chooseReferencePath(p), refPath);
  await waitFor(page, () => window.__testHooks.distillation.state().phase === 'shimmering', 'shimmering phase', 5000).catch(() => {});
  await shot(page, 'distillation-shimmering');

  await waitFor(page, () => window.__testHooks.distillation.state().phase === 'sealed', 'sealed phase', 20000);
  await shot(page, 'distillation-sealed');

  await page.evaluate(() => window.__testHooks.distillation.seal());
  await shot(page, 'distillation-after-seal');

  await clickNav('Main Stage');
  // MainStage remounts and refetches — essence should now be on the shelf.
  await waitFor(page, () => window.__testHooks.stage.state().essenceCount > 0, 'essence on shelf', 10000);
  await shot(page, 'main-stage-with-essence');

  await page.evaluate((p) => window.__testHooks.stage.chooseTargetPath(p), targetPath);
  await waitFor(page, () => !!window.__testHooks.stage.state().targetPath, 'target photo chosen', 5000);
  await shot(page, 'main-stage-target-chosen');

  // Grab the essence id straight from the sidecar rather than the DOM.
  const essences = await page.evaluate(async () => {
    const res = await window.appBridge.sidecarCall('GET', '/essences');
    return res.data.essences;
  });
  const essenceId = essences[0].id;
  console.log('applying essence', essenceId, essences[0].name);

  const applyPromise = page.evaluate((id) => window.__testHooks.stage.applyEssenceById(id), essenceId);
  // The mock apply is fast enough to finish inside one poll interval, so
  // "applying" is best-effort rather than a hard wait.
  await new Promise((r) => setTimeout(r, 150));
  await shot(page, 'main-stage-applying').catch(() => {});
  await applyPromise;
  await shot(page, 'main-stage-applied');

  await clickNav('Media Page');
  await new Promise((r) => setTimeout(r, 1000)); // let the fetch settle
  await shot(page, 'media-page');

  await app.close();
  console.log('DONE');
}

main().catch((err) => {
  console.error('FAILED:', err);
  process.exit(1);
});
