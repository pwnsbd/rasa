#!/usr/bin/env node
// One-time dev setup for the Python sidecar. Creates sidecar/venv and installs
// requirements.txt, then torch: CUDA (cu128) if an NVIDIA GPU is detectable
// via nvidia-smi, CPU otherwise. torch itself must be installed before
// torch.cuda.is_available() can be used, so we can't rely on that check here
// — see sidecar/gpu.py for the runtime detection this setup enables.
const { execSync, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const sidecarDir = path.join(__dirname, '..', 'sidecar');
const venvDir = path.join(sidecarDir, 'venv');
const isWin = process.platform === 'win32';

function venvPython() {
  return isWin ? path.join(venvDir, 'Scripts', 'python.exe') : path.join(venvDir, 'bin', 'python');
}

function run(cmd, args, opts = {}) {
  console.log(`> ${cmd} ${args.join(' ')}`);
  const res = spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  if (res.status !== 0) {
    throw new Error(`Command failed: ${cmd} ${args.join(' ')}`);
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

// Installs the right torch build LAST, with --force-reinstall, after the
// venv's other requirements are in — some packages pull their own pinned
// CPU-only torch as a transitive dep, silently downgrading whatever GPU
// build was there before; installing ours again afterward, forced, wins the
// last-write. Also: cu121 and earlier report torch.cuda.is_available() ==
// True on current-gen "Blackwell" GPUs (RTX 50-series, sm_120, including the
// project's RTX 5070 dev target) but have no compiled kernels for sm_120, so
// real inference crashes with "CUDA error: no kernel image is available for
// execution on the device" — cu128 fixes that (confirmed working on an
// RTX 5070 Ti Laptop GPU as of torch 2.11.0+cu128; see pdfToAudio's
// scripts/setup-sidecar.js for the same finding on a sibling project).
function installTorch(gpu) {
  const python = venvPython();
  if (gpu) {
    run(python, [
      '-m', 'pip', 'install', 'torch', 'torchvision',
      '--index-url', 'https://download.pytorch.org/whl/cu128',
      '--force-reinstall', '--no-deps',
    ]);
  } else {
    run(python, ['-m', 'pip', 'install', 'torch', 'torchvision']);
  }
}

function main() {
  const gpu = hasNvidiaGpu();
  console.log(`NVIDIA GPU detected via nvidia-smi: ${gpu}`);
  if (!gpu) {
    console.log('No GPU detected: falling back to a CPU torch build. Generation will be slow (spec §2.3).');
  }

  if (!fs.existsSync(venvDir)) {
    console.log('Creating virtualenv at sidecar/venv ...');
    run('python', ['-m', 'venv', venvDir]);
  } else {
    console.log('sidecar/venv already exists, skipping creation.');
  }

  const python = venvPython();
  run(python, ['-m', 'pip', 'install', '--upgrade', 'pip']);
  run(python, ['-m', 'pip', 'install', '-r', path.join(sidecarDir, 'requirements.txt')]);
  installTorch(gpu);

  console.log('\nDone. Run `npm run dev` to launch the app, or `sidecar/venv/Scripts/python sidecar/app.py`');
  console.log('to run the sidecar standalone and hit http://127.0.0.1:8843/health directly.');
}

main();
