const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const bootstrap = require('../electron/sidecarBootstrap');
test('runtime readiness tracks dependencies and bootstrap changes', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rasa-test-'));
  try {
    const resourcesPath = path.join(root, 'resources');
    fs.mkdirSync(path.join(resourcesPath, 'sidecar'), { recursive: true });
    const requirements = path.join(resourcesPath, 'sidecar', 'requirements.txt');
    fs.writeFileSync(requirements, 'test==1');
    const dir = bootstrap.runtimeDir(root);
    fs.mkdirSync(dir);
    const python = bootstrap.runtimePython(root);
    fs.writeFileSync(python, '');
    const fingerprint = crypto.createHash('sha256').update(fs.readFileSync(requirements)).update(fs.readFileSync(require.resolve('../electron/sidecarBootstrap'))).digest('hex');
    fs.writeFileSync(path.join(dir, '.rasa-runtime-ready'), fingerprint);
    assert.equal(await bootstrap.ensureSidecarRuntime({ userDataRoot: root, resourcesPath }), python);
    fs.writeFileSync(requirements, 'test==2');
    await assert.rejects(bootstrap.ensureSidecarRuntime({ userDataRoot: root, resourcesPath }), /Bundled Python runtime not found/);
    fs.writeFileSync(requirements, 'test==1');
    fs.unlinkSync(python);
    await assert.rejects(bootstrap.ensureSidecarRuntime({ userDataRoot: root, resourcesPath }), /Bundled Python runtime not found/);
  } finally {
    if (path.dirname(root) !== os.tmpdir() || !path.basename(root).startsWith('rasa-test-')) throw Error('Unsafe cleanup path');
    fs.rmSync(root, { recursive: true, force: true });
  }
});
