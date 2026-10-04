# Windows release candidate

Build with npm run dist. Output: release/Rasa-Setup-<version>-x64.exe. Current public release: v0.2.0 (2026-10-04).

The installer creates shortcuts and launches Rasa. Users do not need Node, Python, pip, or developer tools. First launch still requires internet: it installs Python dependencies and downloads approximately 13 GB of models, in addition to runtime packages. This is an online-setup installer, not an offline bundle.

Startup now reports failures with a retry button, prevents duplicate application instances, cleans up setup processes on quit, explicitly resolves backend modules in embedded Python, and rebuilds the private runtime when dependency/bootstrap definitions change. App data is retained on uninstall.

Done for v0.2.0 (2026-10-04):
- Fresh Windows account install from the public download link, first-run setup, model download, distill/restyle/save: passed (Pawan).
- Transitive Python dependencies and torch versions locked (`sidecar/constraints.txt`, `sidecar/torch-versions.json`).
- Disk-space preflight before model download.

Still open:
- Test interrupted setup, offline restart after preparation, upgrade, uninstall/reinstall, and non-ASCII user paths.
- A user-selectable storage directory (PLAN #4).
- Review Electron/dependency security updates and model distribution notices.
- Configure a trusted publisher signing certificate before broad distribution. Do not treat an unsigned build as a signed public release.

Validation performed is recorded in the task response; building an installer alone is not a clean-machine acceptance test.
