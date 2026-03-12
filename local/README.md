# Local plugins and services

Place your own command plugins and service plugins here so they stay separate from bot-provided code.

- **local/commands/** — Python files each defining a command (subclass of `BaseCommand`). Loaded after built-in commands; duplicate names are skipped.
- **local/service_plugins/** — Python files each defining a background service (subclass of `BaseServicePlugin`). Loaded after built-in services; duplicate names are skipped.
- **local/config.ini** — Optional. Merged with main `config.ini`; use it for sections and options for your local plugins and services.

See **docs/local-plugins.md** (or the "Local plugins and services" section in the docs) for how to write a minimal plugin and configure it.
