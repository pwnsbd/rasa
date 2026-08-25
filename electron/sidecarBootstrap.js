// First-run setup for a PACKAGED install's sidecar Python environment.
// Dev (`npm run dev`) never touches this — it always uses sidecar/venv,
// built ahead of time by `npm run sidecar:setup`. A packaged install has no
// such venv (and can't assume the end user has Python installed at all —
// see scripts/fetch-embed-python.js's docstring), so this module builds one
// on first launch instead, from the embeddable Python bundled into the
// installer as extraResources (package.json's build.extraResources).
//
// The bundled resources/python-embed/ is already a complete, pip-ready
// interpreter by the time it's packaged (fetch-embed-python.js does the
// extraction/patching/pip-bootstrap once, at build-prep time, on the
// developer's machine, verified working there) — so this module's only
// job at the end user's actual first run is: copy it into a writable
// per-user location, then `pip install` the sidecar's own requirements +
// the right torch build for their machine. That copy step matters: the
// bundled copy lives under the installed app's own resources, which a
// per-machine install could put somewhere requiring admin rights to
// write into (site-packages) even if the initial install didn't; a
// per-user copy under userData is always writable, and survives an app
// update independent of whatever the update replaces under resourcesPath.
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const READY_MARKER = '.rasa-runtime-ready';

function runtimeDir(userDataRoot) {
  return path.join(userDataRoot, 'sidecar-runtime');
}

function runtimePython(userDataRoot) {
  return path.join(runtimeDir(userDataRoot), 'python.exe');
}

function isReady(userDataRoot) {
  return fs.existsSync(path.join(runtimeDir(userDataRoot), READY_MARKER));
}

function copyRecursive(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyRecursive(s, d);
    } else {
      fs.copyFileSync(s, d);
    }
  }
}

function hasNvidiaGpu() {
  try {
    execSync('nvidia-smi', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

// Runs a command, streaming stdout/stderr lines to `onProgress` (for the
// renderer-facing status text) instead of inheriting stdio the way
// scripts/setup-sidecar.js's dev-only equivalent does — this runs inside
// the packaged app, not a terminal a developer is watching.
function run(cmd, args, onProgress) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { windowsHide: true });
    child.stdout.on('data', (d) => onProgress?.(d.toString()));
    child.stderr.on('data', (d) => onProgress?.(d.toString()));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

// installTorch mirrors scripts/setup-sidecar.js's own logic exactly (same
// cu128 reasoning — see that file's comment on current-gen "Blackwell" GPUs
// needing cu128 specifically, not an older CUDA build) — duplicated rather
// than shared because that script is a dev-only CLI entry point (uses
// spawnSync + inherited stdio) and this one needs the streaming/async shape
// above to report progress to a window instead of a terminal.
async function installTorch(python, gpu, onProgress) {
  if (gpu) {
    await run(
      python,
      ['-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128', '--force-reinstall', '--no-deps'],
      onProgress,
    );
  } else {
    await run(python, ['-m', 'pip', 'install', 'torch', 'torchvision'], onProgress);
  }
}

/**
 * Sets up (or confirms) the packaged sidecar's Python runtime.
 * @param {object} opts
 * @param {string} opts.userDataRoot - writable per-user app-data root (Electron's app.getPath('userData'))
 * @param {string} opts.resourcesPath - process.resourcesPath (where extraResources landed)
 * @param {(status: {step: string, detail?: string}) => void} [opts.onProgress]
 * @returns {Promise<string>} path to the ready-to-use python.exe
 */
async function ensureSidecarRuntime({ userDataRoot, resourcesPath, onProgress }) {
  const dir = runtimeDir(userDataRoot);
  const python = runtimePython(userDataRoot);
  const report = (step, detail) => onProgress?.({ step, detail });

  if (isReady(userDataRoot) && fs.existsSync(python)) {
    return python;
  }

  const bundledPython = path.join(resourcesPath, 'python-embed');
  if (!fs.existsSync(path.join(bundledPython, 'python.exe'))) {
    throw new Error(
      `Bundled Python runtime not found at ${bundledPython} — this build wasn't packaged with ` +
        '"npm run dist" (which runs scripts/fetch-embed-python.js first). See package.json.',
    );
  }

  report('Setting up Rasa for the first time…', 'Copying Python runtime');
  fs.rmSync(dir, { recursive: true, force: true }); // clean slate — a half-finished previous attempt shouldn't linger
  copyRecursive(bundledPython, dir);

  report('Setting up Rasa for the first time…', 'Installing dependencies (this can take a few minutes)');
  await run(python, ['-m', 'pip', 'install', '-r', path.join(resourcesPath, 'sidecar', 'requirements.txt')], (line) =>
    report('Setting up Rasa for the first time…', line.trim() || undefined),
  );

  const gpu = hasNvidiaGpu();
  report('Setting up Rasa for the first time…', gpu ? 'Installing GPU (CUDA) support' : 'Installing CPU support (no GPU detected — generation will be slow)');
  await installTorch(python, gpu, (line) => report('Setting up Rasa for the first time…', line.trim() || undefined));

  fs.writeFileSync(path.join(dir, READY_MARKER), new Date().toISOString());
  report('Setup complete');
  return python;
}

module.exports = { ensureSidecarRuntime, isReady, runtimePython, runtimeDir };
