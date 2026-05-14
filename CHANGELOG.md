# Changelog

## [0.1.1-alpha] - 2026-05-14

### Security
- Stop reading exchange and Gemini API keys from `config.json`; secrets are loaded from environment variables only.
- Enforce bounded IPC header/body reads with oversized request rejection.
- Add whitelist and range validation for runtime config updates.

### Fixed
- Consolidate `TradingSignal` on one canonical dataclass.
- Keep AI signal sizing data in metadata so pipeline contracts stay compatible.

### Changed
- Refresh Money Machine desktop logo/icons and Shadow Mode dashboard UI.
- Add Vercel and Cloudflare Pages security headers for the dashboard build.

## [0.1.0-alpha] - 2026-05-14

### Added
- Initial alpha release tag for AlphaAxiom Money Machine.
