# Backwards Compatibility Policy

## Public API
The public API consists of:
1. Symbols exported from `stackformer.__all__`.
2. Constructor signatures and behavior for major model classes in `stackformer.models`.
3. Utility behavior for documented generation and trainer entry points.

## Versioning rules
- **MAJOR**: incompatible API changes.
- **MINOR**: backward-compatible feature additions.
- **PATCH**: backward-compatible bug fixes and docs-only changes.

## Deprecation policy
- Deprecated APIs are warned in one MINOR release before removal.
- Removal happens only in a subsequent MAJOR release.

## Compatibility target
- We target Python versions declared in `pyproject.toml`.
- Optional features may require optional extras (e.g., `train`).
