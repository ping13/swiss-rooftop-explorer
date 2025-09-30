# Repository Guidelines

## Project Structure & Module Organization
- `scripts/`: Python ETL pipeline that ingests Swiss geodata and writes intermediate parquet chunks.
- `assets/`: Raw downloads and generated spatial files; keep large datasets out of Git and regenerate via Make targets.
- `web/`: Vite-based frontend using MapLibre and DuckDB-WASM (`src/main.js`, `src/addresses.js`, `src/style.css`).
- `web/public/`: Published artifacts such as parquet extracts and `buildings.pmtiles`; treat as build output.
- `qgis/`: Project files and helpers for manual bridge inspection; `Attic/` holds experiments kept for reference.

## Build, Test, and Development Commands
- `source init.sh`: Activates the conda environment expected by the Makefile before running data jobs.
- `make download | create_addresses | create_buildings | create_bridges`: Stage the ETL pipeline; run sequentially when refreshing datasets.
- `make pmtiles`: Rebuilds the PMTiles used by the web map after geometry changes.
- `cd web && npm install`: Install UI dependencies; `npm run dev` starts the local map, `npm run build` prepares `web/dist` for deploy, `npm run preview` smoke-tests the build.
- `make devserver`: Convenience target that launches the frontend with the correct environment.

## Coding Style & Naming Conventions
- Python adheres to PEP 8 with 4-space indents; use descriptive snake_case for functions/modules and keep heavy processing in `scripts/`.
- JavaScript uses ES modules with 4-space indents; centralize shared helpers (e.g., DuckDB access) and keep constants uppercase (`PMTILES_URL`).
- Keep CSS selectors grouped by component in `web/src/style.css`; avoid inline styles in the markup.
- Generated files follow dataset-driven names (`*_Building_solid.parquet`); do not manually rename outputs or checked-in assets.

## Testing Guidelines
- No formal CI yet; lean on targeted smoke checks before publishing.
- For Python ETL updates, run scripts against small chunks (`--chunk 1000`) and validate parquet schema changes before full runs.
- For frontend work, use `npm run dev` to verify ZIP search, map highlighting, and bridge overlays; capture console warnings.
- Log notable Make runs and resulting artifact sizes when sharing results.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) as used in the recent history.
- Keep commits scoped to a single concern (e.g., frontend vs. ETL) and document any regenerated artifacts in the message body.
- Pull requests should summarize the change, link related issues, and include screenshots or map URLs demonstrating UI updates.
- Call out manual follow-up steps (`make publish_*`) so reviewers know whether additional actions are required.
