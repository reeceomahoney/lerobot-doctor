"""Episode trimming — remove idle/static frames from episodes.

Detects and removes frames where the robot isn't moving (common at
the start/end of teleoperation recordings, and as internal stalls
where the operator hesitates). These idle frames teach the policy to
"do nothing" which causes stuck behaviors at inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class TrimResult:
    episodes_trimmed: int
    frames_removed: int
    episodes_removed: int
    details: list[str]


def trim_dataset(
    root: Path,
    action_threshold: float = 0.01,
    min_active_frames: int = 10,
    trim_start: bool = True,
    trim_end: bool = True,
    min_frozen_run: int = 5,
    remove_fully_static: bool = True,
    dry_run: bool = False,
) -> TrimResult:
    """Trim idle frames from episodes.

    Args:
        root: Dataset root path
        action_threshold: Frames with action std below this are "idle"
        min_active_frames: Minimum active frames to keep an episode
        trim_start: Remove idle frames at start of episodes
        trim_end: Remove idle frames at end of episodes
        min_frozen_run: Also drop internal runs of idle frames whose length
            is >= this value. Set to 0 to only trim start/end.
        remove_fully_static: Remove episodes that are entirely static
        dry_run: Only report, don't modify
    """
    result = TrimResult(episodes_trimmed=0, frames_removed=0, episodes_removed=0, details=[])

    data_dir = root / "data"
    if not data_dir.exists():
        return result

    parquet_files = sorted(data_dir.rglob("*.parquet"))

    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            if "episode_index" not in table.column_names:
                continue

            # Find action columns
            action_cols = [c for c in table.column_names if c.startswith("action")]
            if not action_cols:
                continue

            ep_col = table.column("episode_index").to_pylist()
            episodes = {}
            for i, ep in enumerate(ep_col):
                if ep not in episodes:
                    episodes[ep] = []
                episodes[ep].append(i)

            rows_to_keep = []
            for ep_idx in sorted(episodes.keys()):
                row_indices = episodes[ep_idx]

                # Get action values for this episode. Handles both scalar columns
                # (action.x, action.y, ...) and a single vector column where each
                # row is a list (the v3 default `action` column).
                actions = []
                for col in action_cols:
                    try:
                        vals = np.array([table.column(col)[i].as_py() for i in row_indices], dtype=np.float64)
                    except (ValueError, TypeError):
                        continue
                    if vals.ndim == 1:
                        actions.append(vals.reshape(-1, 1))
                    elif vals.ndim == 2:
                        actions.append(vals)

                if not actions:
                    rows_to_keep.extend(row_indices)
                    continue

                action_matrix = np.concatenate(actions, axis=1)

                # Compute per-frame "activity" as action change magnitude
                if len(action_matrix) < 2:
                    rows_to_keep.extend(row_indices)
                    continue

                diffs = np.abs(np.diff(action_matrix, axis=0))
                activity = np.concatenate([[0], diffs.mean(axis=1)])
                is_active = activity > action_threshold

                # Find first and last active frames
                active_indices = np.where(is_active)[0]

                if len(active_indices) < min_active_frames:
                    if remove_fully_static:
                        result.episodes_removed += 1
                        result.frames_removed += len(row_indices)
                        result.details.append(f"Episode {ep_idx}: removed (fully static)")
                        continue
                    else:
                        rows_to_keep.extend(row_indices)
                        continue

                start = active_indices[0] if trim_start else 0
                end = active_indices[-1] + 1 if trim_end else len(row_indices)

                trimmed_indices = row_indices[start:end]

                # Drop internal runs of idle frames whose length is >= min_frozen_run
                if min_frozen_run > 0 and len(trimmed_indices) > 0:
                    idle = (~is_active[start:end]).astype(np.int8)
                    edges = np.diff(np.concatenate([[0], idle, [0]]))
                    run_starts = np.where(edges == 1)[0]
                    run_ends = np.where(edges == -1)[0]
                    keep = np.ones(len(trimmed_indices), dtype=bool)
                    for s, e in zip(run_starts, run_ends):
                        if e - s >= min_frozen_run:
                            keep[s:e] = False
                    trimmed_indices = [trimmed_indices[i] for i in np.where(keep)[0]]

                n_removed = len(row_indices) - len(trimmed_indices)

                if n_removed > 0:
                    result.episodes_trimmed += 1
                    result.frames_removed += n_removed
                    result.details.append(
                        f"Episode {ep_idx}: trimmed {n_removed} idle frames "
                        f"({len(row_indices)} → {len(trimmed_indices)})"
                    )

                rows_to_keep.extend(trimmed_indices)

            # Write trimmed table. Per-row `frame_index` and `timestamp` are
            # left at their original values so video frame seek (which uses
            # episode `from_timestamp` + per-row `timestamp`) still resolves
            # to the correct frame in the unmodified video files. The global
            # `index` column is rebuilt in `_update_metadata_after_trim`.
            if len(rows_to_keep) < len(ep_col) and not dry_run:
                trimmed_table = table.take(rows_to_keep)
                pq.write_table(trimmed_table, pf)

        except Exception as e:
            result.details.append(f"Error processing {pf.name}: {e}")

    # Update metadata
    if not dry_run and (result.frames_removed > 0 or result.episodes_removed > 0):
        _update_metadata_after_trim(root)

    return result


def _update_metadata_after_trim(root: Path):
    """Update info.json, the data `index` column, and episodes metadata after
    trimming.

    Preserves per-row `timestamp` and `frame_index`, and per-episode video
    `from_timestamp`/`to_timestamp` — the video files themselves are not
    modified, so the original timestamps remain the correct seek keys.

    Rebuilds:
      - data `index` column (contiguous 0..N across kept rows)
      - episodes `length`, `dataset_from_index`, `dataset_to_index`
      - info.json `total_frames`, `total_episodes`

    Episodes that were entirely removed (fully-static) are dropped from
    the episodes metadata.
    """
    data_dir = root / "data"
    if not data_dir.exists():
        return

    parquet_files = sorted(data_dir.rglob("*.parquet"))

    # Pass 1: rebuild contiguous global `index`; record per-episode row range.
    global_idx = 0
    episode_bounds: dict[int, tuple[int, int]] = {}
    for pf in parquet_files:
        table = pq.read_table(pf)
        ep_col = table.column("episode_index").to_pylist()
        n = len(ep_col)
        if n == 0:
            continue
        new_index = list(range(global_idx, global_idx + n))
        for offset, ep in enumerate(ep_col):
            row = global_idx + offset
            if ep not in episode_bounds:
                episode_bounds[ep] = (row, row + 1)
            else:
                episode_bounds[ep] = (episode_bounds[ep][0], row + 1)
        global_idx += n
        if "index" in table.column_names:
            table = table.set_column(
                table.column_names.index("index"),
                "index",
                pa.array(new_index),
            )
            pq.write_table(table, pf)

    # Pass 2: update episodes metadata in place, preserving all other columns.
    episodes_dir = root / "meta" / "episodes"
    if episodes_dir.exists():
        for mf in sorted(episodes_dir.rglob("*.parquet")):
            meta_table = pq.read_table(mf)
            if "episode_index" not in meta_table.column_names:
                continue
            ep_col = meta_table.column("episode_index").to_pylist()
            keep_mask = [ep in episode_bounds for ep in ep_col]
            if not any(keep_mask):
                mf.unlink()
                continue
            meta_table = meta_table.filter(pa.array(keep_mask))
            ep_col = meta_table.column("episode_index").to_pylist()

            updates = {
                "dataset_from_index": [episode_bounds[ep][0] for ep in ep_col],
                "dataset_to_index": [episode_bounds[ep][1] for ep in ep_col],
                "length": [episode_bounds[ep][1] - episode_bounds[ep][0] for ep in ep_col],
            }
            for col, vals in updates.items():
                if col in meta_table.column_names:
                    meta_table = meta_table.set_column(
                        meta_table.column_names.index(col), col, pa.array(vals)
                    )
            pq.write_table(meta_table, mf)

    # Pass 3: info.json totals.
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        import json as _json
        info = _json.loads(info_path.read_text())
        info["total_frames"] = global_idx
        info["total_episodes"] = len(episode_bounds)
        info_path.write_text(_json.dumps(info, indent=2))
