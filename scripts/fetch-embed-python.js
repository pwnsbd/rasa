#!/usr/bin/env node
// Build-prep step (npm run dist:prep, or `npm run dist` runs it automatically):
// downloads Python's official Windows "embeddable" distribution into
// resources/python-embed/ so electron-builder can bundle it as extraResources
// (see package.json's build.extraResources). This is NOT a full Python
// install — it's a minimal, redistributable interpreter (~10-16MB) meant
// exactly for embedding inside another application, which is what lets a
// packaged Rasa install set up its own sidecar Python environment on first
// run (electron/sidecarBootstrap.js) without requiring the end user to have
// Python installed at all.
//
// Pinned to match the version the dev sidecar/venv actually uses (checked via
// `sidecar/venv/Scripts/python.exe --version` at the time this was written:
// 3.12.10) — torch/diffusers wheel availability can lag the very latest
// Python minor version, so matching the already-proven dev version is safer
// than always grabbing the newest release.
//
// Not committed to git (resources/ is gitignored) — this script is the
// reproducible source of truth instead, run once before packaging.
const https = require('https');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const PYTHON_VERSION = '3.12.10';
const ZIP_NAME = `python-${PYTHON_VERSION}-embed-amd64.zip`;
const DOWNLOAD_URL = `https://www.python.org/ftp/python/${PYTHON_VERSION}/${ZIP_NAME}`;

const outDir = path.join(__dirname, '..', 'resources', 'python-embed');
const zipPath = path.join(__dirname, '..', 'resources', ZIP_NAME);

function download(url, dest) {
  return new Promise((resolve, reject) => {
    console.log(`> downloading ${url}`);
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        if (res.statusCode === 302 || res.statusCode === 301) {
          file.close();
          fs.unlinkSync(dest);
          return download(res.headers.location, dest).then(resolve, reject);
        }
        if (res.statusCode !== 200) {
          file.close();
          fs.unlinkSync(dest);
          return reject(new Error(`Download failed: HTTP ${res.statusCode} for ${url}`));
        }
        res.pipe(file);
        file.on('finish', () => file.close(resolve));
      })
      .on('error', (err) => {
        fs.unlink(dest, () => reject(err));
      });
  });
}

function extract(zip, dest) {
  console.log(`> extracting to ${dest}`);
  fs.mkdirSync(dest, { recursive: true });
  // Expand-Archive, not a new npm zip dependency — this build step only
  // ever needs to run on Windows (it's producing a Windows embeddable
  // interpreter for a Windows NSIS build), so PowerShell is always present.
  const res = spawnSync(
    'powershell.exe',
    ['-NoProfile', '-Command', `Expand-Archive -Path "${zip}" -DestinationPath "${dest}" -Force`],
    { stdio: 'inherit' },
  );
  if (res.status !== 0) {
    throw new Error('Expand-Archive failed');
  }
}

// The embeddable distribution ships with `import site` commented out in its
// ._pth file, which — among other things — disables pip's ability to find
// site-packages at all once installed. Uncommenting it is the documented
// fix (see python.org's embeddable-package notes) and is required before
// get-pip.py + `pip install` will work correctly.
function enableSitePackages(dest) {
  const pthFile = fs.readdirSync(dest).find((f) => f.endsWith('._pth'));
  if (!pthFile) {
    throw new Error('Expected a ._pth file in the extracted embeddable Python, found none');
  }
  const pthPath = path.join(dest, pthFile);
  const content = fs.readFileSync(pthPath, 'utf8');
  const patched = content.replace(/^#\s*import site/m, 'import site');
  fs.writeFileSync(pthPath, patched);
  console.log(`> enabled site-packages in ${pthFile}`);
}

// Bootstraps pip into the embeddable interpreter at build-prep time, not at
// the end user's first run — so the packaged resources/python-embed/ is a
// complete, ready-to-`pip install` environment, and sidecarBootstrap.js's
// actual first-run step (installing sidecar/requirements.txt + torch) is
// the only thing that needs network access on the end user's machine.
// Verified for real while writing this script: extraction, the ._pth
// patch, and this pip bootstrap all confirmed working (a plain `pip
// install requests` succeeded and imported cleanly) before this file
// reached its current form.
function bootstrapPip(dest) {
  const getPipPath = path.join(dest, 'get-pip.py');
  console.log('> bootstrapping pip');
  const pythonExe = path.join(dest, 'python.exe');
  const res = spawnSync(pythonExe, [getPipPath, '--no-warn-script-location'], { stdio: 'inherit' });
  if (res.status !== 0) {
    throw new Error('get-pip.py failed');
  }
  fs.unlinkSync(getPipPath); // only needed once, at build-prep time
}

async function main() {
  if (fs.existsSync(outDir) && fs.existsSync(path.join(outDir, 'python.exe'))) {
    console.log('resources/python-embed already present, skipping (delete it to force a re-fetch).');
    return;
  }
  fs.mkdirSync(path.dirname(zipPath), { recursive: true });
  await download(DOWNLOAD_URL, zipPath);
  extract(zipPath, outDir);
  enableSitePackages(outDir);
  fs.unlinkSync(zipPath);
  await download('https://bootstrap.pypa.io/get-pip.py', path.join(outDir, 'get-pip.py'));
  bootstrapPip(outDir);
  console.log('Done. resources/python-embed is a ready-to-use, pip-enabled Python runtime.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
