# Null and Missing-Data Handling in Simulation Tools

This folder contains traffic replay and visualization tools. Their nullity handling is implemented as input checks, empty-data handling, and fallbacks for optional dependencies—not as a single shared null-value framework.

## Existing behavior

- **Replay parsing:** `frontend_replay/replay_simu.py` skips blank lines, lines without the expected `;` separator, and vehicle or traffic-light entries whose token counts do not match the expected format. A replay with no valid frames is reported and the visualizer exits rather than starting an empty animation.
- **Analysis dependencies and data:** `frontend_replay/current/currnet.py` reports missing NumPy as a required dependency, disables SciPy-dependent model fitting when SciPy is absent, and disables plotting when Matplotlib is unavailable. Growth analysis returns an error result when a replay file cannot be read or contains no frames; streaming analysis reports when no usable data was parsed.
- **Missing routes and paths:** the grid replay skips flows with empty routes and warns or removes a vehicle when the required lane path is unavailable. Unmapped signal links are omitted from the rendered light data.

## Practical interpretation

These behaviors let the tools continue when optional display helpers or individual malformed records are absent, while reporting some missing-file, empty-data, and unsupported-format conditions. A skipped record is not the same as a numeric zero: skipped input may reduce the data represented in a visualization or analysis. Check the console output and source data when completeness matters.

The checks are format-specific. Some code paths still expect required JSON keys, valid numeric values, known road or intersection IDs, and nonempty geometry. The utilities do not guarantee that all null-like values (`null`, missing keys, empty strings, `NaN`, or `None`) are detected consistently. Validate input files before relying on results, and treat incomplete data as incomplete rather than silently interpreting it as zero.

## Related files

- `frontend_replay/replay_simu.py` — replay and road-network parsing, record skipping, and empty-frame check.
- `frontend_replay/current/currnet.py` — replay data ingestion and dependency/data availability handling.
- `tools_generator_grid_replay/grid_data/grid_9_9/project/cityflow_to_sumo_converter.py` — supported input formats and JSON/flow handling.
- `tools_generator_grid_replay/visu.py` — grid geometry and traffic-flow replay visualization.

## Validation recommendations

For new inputs, verify file existence, JSON syntax, expected top-level shapes and required fields, numerical finiteness, and route/road references before simulation. When records are skipped, log counts and reasons so downstream summaries make the reduced coverage visible. Avoid replacing missing observations with zero unless zero is a valid measured value for that field.
