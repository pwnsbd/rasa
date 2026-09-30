# Windows release candidate

Build with npm run dist. Output: release/Rasa-Setup-0.1.0-x64.exe.

The installer creates shortcuts and launches Rasa. Users do not need Node, Python, pip, or developer tools. First launch still requires internet: it installs Python dependencies and downloads approximately 13 GB of models, in addition to runtime packages. This is an online-setup installer, not an offline bundle.

Startup now reports failures with a retry button, prevents duplicate application instances, cleans up setup processes on quit, explicitly resolves backend modules in embedded Python, and rebuilds the private runtime when dependency/bootstrap definitions change. App data is retained on uninstall.

Before public release:
- Test the installer in a fresh Windows account/machine without Python or Node, including a supported NVIDIA GPU and actual extract/apply/export operations.
- Test interrupted setup, offline restart after preparation, upgrade, uninstall/reinstall, and non-ASCII user paths.
- Verify sufficient disk space for model downloads, runtime, cache, and generated media. A user-selectable storage directory and a disk-space preflight remain desirable.
- Review and lock transitive Python dependencies and torch versions; currently they can resolve differently on future installs.
- Review Electron/dependency security updates and model distribution notices.
- Configure a trusted publisher signing certificate before broad distribution. Do not treat an unsigned build as a signed public release.

Validation performed is recorded in the task response; building an installer alone is not a clean-machine acceptance test.
