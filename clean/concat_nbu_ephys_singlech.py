import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


FILE_PREFIX = "spectra_highres_spectra_"
FILE_GLOB = "*/LFP/clean/spectra_highres_spectra_*.pkl"
BASE_DIR = Path("/mnt/projectworlds/TRBD-53761/TRBD001/NBU")  # TRBD001
#/Volumes/projectworlds/TRBD-53761/TRBD001/NBU/2025-06-10_2025-06-11/LFP/clean/spectra_highres_spectra_ONE_THREE_LEFT.pkl
OUTPUT_DIR = Path("clean/nbu_concat_single_channel_TRBD001")  # TRBD001
MAX_GAP_DAYS = 4
PREFERRED_CONTACTS = ("ONE_THREE", "ZERO_TWO", "ZERO_THREE")
VALID_SIDES = ("LEFT", "RIGHT")
VISIT_SPLIT_START_DATE = pd.Timestamp("2025-06-03")
VISIT_SPLIT_TIMESTAMP = pd.Timestamp("2025-06-04 22:56:14-05:00")
VISIT_SPLIT_CUTOFF_DAY = VISIT_SPLIT_TIMESTAMP.tz_localize(None).normalize()


@dataclass
class FileRecord:
    path: Path
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    channel: str
    contacts: str
    side: str


@dataclass
class MergedRecord:
    date: pd.Timestamp
    side: str
    payload: dict[str, Any]
    source_channels: list[str]
    source_paths: list[str]


def parse_file_record(path: Path) -> FileRecord:
    # Expected date location: <base>/<YYYY-MM-DD>/rawlfp/raw/<file>.pkl
    date_str = path.parent.parent.parent.name
    date_start, date_end = date_str.split("_")
    date_start = pd.Timestamp(date_start)
    date_end = pd.Timestamp(date_end)

    stem = path.stem
    if not stem.startswith(FILE_PREFIX):
        raise ValueError(f"Unexpected filename format: {path.name}")

    channel = stem[len(FILE_PREFIX) :]
    parts = channel.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected channel format: {channel}")

    side = parts[-1]
    contacts = "_".join(parts[:-1])

    if side not in VALID_SIDES:
        raise ValueError(f"Unsupported side in channel {channel}")

    return FileRecord(path=path, date_start=date_start, date_end=date_end, channel=channel, contacts=contacts, side=side)


def split_by_gap(records: list[Any], max_gap_days: int) -> list[list[Any]]:
    if not records:
        return []

    groups: list[list[FileRecord]] = []
    current_group: list[FileRecord] = [records[0]]

    for rec in records[1:]:
        prev = current_group[-1]
        gap_days = (rec.date - prev.date).days
        if gap_days <= max_gap_days:
            current_group.append(rec)
        else:
            groups.append(current_group)
            current_group = [rec]

    groups.append(current_group)
    return groups


def concat_dict_payload(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    bucket: dict[str, list[Any]] = defaultdict(list)

    for payload in dicts:
        for key, value in payload.items():
            bucket[key].append(value)

    for key, values in bucket.items():
        first = values[0]

        if isinstance(first, pd.DataFrame):
            frames = [v for v in values if isinstance(v, pd.DataFrame)]
            out = pd.concat(frames).sort_index()
            out = out[~out.index.duplicated(keep="first")]
            merged[key] = out
        elif isinstance(first, pd.Series):
            series_list = [v for v in values if isinstance(v, pd.Series)]
            out = pd.concat(series_list).sort_index()
            out = out[~out.index.duplicated(keep="first")]
            merged[key] = out
        else:
            # Keep the most recent non-tabular metadata value.
            merged[key] = values[-1]

    return merged


def load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _order_channels_by_preference(channels: list[str], side: str) -> list[str]:
    ordered: list[str] = []
    remaining = sorted(set(channels))

    for preferred in PREFERRED_CONTACTS:
        preferred_channel = f"{preferred}_{side}"
        if preferred_channel in remaining:
            ordered.append(preferred_channel)
            remaining.remove(preferred_channel)

    ordered.extend(remaining)
    return ordered


def _merge_frame_by_preference(
    frames_by_channel: dict[str, pd.DataFrame],
    channel_order: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if not frames_by_channel:
        return pd.DataFrame(), pd.Series(dtype="object")

    index_union = sorted(set().union(*[frame.index for frame in frames_by_channel.values()]))
    columns_union = sorted(set().union(*[frame.columns for frame in frames_by_channel.values()]))

    merged = pd.DataFrame(index=index_union, columns=columns_union)
    source_channel = pd.Series(index=index_union, dtype="object")

    for channel in channel_order:
        frame = frames_by_channel.get(channel)
        if frame is None:
            continue
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="first")]
        frame = frame.reindex(index_union)

        available = frame.notna().any(axis=1)
        take = available & source_channel.isna()
        if not take.any():
            continue

        merged.loc[take, frame.columns] = frame.loc[take, frame.columns]
        source_channel.loc[take] = channel

    merged = merged.sort_index()
    source_channel = source_channel.sort_index()
    return merged, source_channel


def merge_candidates_for_day(candidates: list[FileRecord]) -> MergedRecord:
    if not candidates:
        raise ValueError("Expected at least one candidate record")

    date = candidates[0].date_start
    side = candidates[0].side

    payloads_by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths_by_channel: dict[str, list[str]] = defaultdict(list)
    for rec in candidates:
        payloads_by_channel[rec.channel].append(load_pickle(rec.path))
        paths_by_channel[rec.channel].append(str(rec.path))

    merged_payload_by_channel: dict[str, dict[str, Any]] = {}
    for channel, payloads in payloads_by_channel.items():
        merged_payload_by_channel[channel] = concat_dict_payload(payloads)

    channel_order = _order_channels_by_preference(list(merged_payload_by_channel.keys()), side)
    candidate_channels = sorted(merged_payload_by_channel.keys())
    if len(candidate_channels) > 1:
        print(
            f"{date.date()} {side}: merge order {channel_order}; candidates {candidate_channels}"
        )

    merged_payload: dict[str, Any] = {}
    all_keys = sorted(
        set().union(*[payload.keys() for payload in merged_payload_by_channel.values()])
    )

    for key in all_keys:
        frames_by_channel: dict[str, pd.DataFrame] = {}
        for channel in channel_order:
            payload = merged_payload_by_channel.get(channel, {})
            value = payload.get(key)
            if isinstance(value, pd.DataFrame):
                frame = value.copy()
                if key == "df_features":
                    frame = frame.drop(columns=["source_channel"], errors="ignore")
                frames_by_channel[channel] = frame

        if frames_by_channel:
            merged_frame, source_channel = _merge_frame_by_preference(frames_by_channel, channel_order)
            if key == "df_features":
                merged_frame = merged_frame.copy()
                merged_frame["source_channel"] = source_channel
            merged_payload[key] = merged_frame
            continue

        series_by_channel: dict[str, pd.Series] = {}
        for channel in channel_order:
            payload = merged_payload_by_channel.get(channel, {})
            value = payload.get(key)
            if isinstance(value, pd.Series):
                series_by_channel[channel] = value.copy()

        if series_by_channel:
            # Concatenate in preference order so higher-preference channels keep duplicates first.
            ordered_series = [series_by_channel[ch] for ch in channel_order if ch in series_by_channel]
            out = pd.concat(ordered_series).sort_index()
            out = out[~out.index.duplicated(keep="first")]
            merged_payload[key] = out
            continue

        for channel in channel_order:
            payload = merged_payload_by_channel.get(channel, {})
            if key in payload:
                merged_payload[key] = payload[key]
                break

    source_channels = [ch for ch in channel_order if ch in merged_payload_by_channel]
    source_paths: list[str] = []
    for ch in source_channels:
        source_paths.extend(sorted(paths_by_channel.get(ch, [])))

    return MergedRecord(
        date=date,
        side=side,
        payload=merged_payload,
        source_channels=source_channels,
        source_paths=source_paths,
    )


def build_output_name(side: str, payload: dict[str, Any], group_idx: int) -> str:
    start = payload["df_spec"].index[0].strftime("%Y-%m-%d")
    end = payload["df_spec"].index[-1].strftime("%Y-%m-%d")
    merged_channel = f"ONEORZERO_THREE_{side}"
    return f"{FILE_PREFIX}{merged_channel}_visit{group_idx}_{start}_to_{end}.pkl"


def build_output_name_with_suffix(
    side: str,
    payload: dict[str, Any],
    group_idx,
    suffix: str,
) -> str:
    base_name = build_output_name(side, payload, group_idx)
    if not suffix:
        return base_name
    return f"{Path(base_name).stem}_{suffix}.pkl"


def _align_split_timestamp(index: pd.DatetimeIndex, split_ts: pd.Timestamp) -> pd.Timestamp:
    if index.tz is None and split_ts.tz is not None:
        return split_ts.tz_localize(None)
    if index.tz is not None and split_ts.tz is None:
        return split_ts.tz_localize(index.tz)
    if index.tz is not None and split_ts.tz is not None:
        return split_ts.tz_convert(index.tz)
    return split_ts


def split_payload_by_timestamp(
    payload: dict[str, Any],
    split_ts: pd.Timestamp,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    before_payload: dict[str, Any] = {}
    after_payload: dict[str, Any] = {}
    before_has_rows = False
    after_has_rows = False

    for key, value in payload.items():
        if isinstance(value, (pd.DataFrame, pd.Series)) and isinstance(value.index, pd.DatetimeIndex):
            split_cmp = _align_split_timestamp(value.index, split_ts)
            before_value = value.loc[value.index <= split_cmp]
            after_value = value.loc[value.index > split_cmp]
            before_payload[key] = before_value
            after_payload[key] = after_value
            before_has_rows = before_has_rows or not before_value.empty
            after_has_rows = after_has_rows or not after_value.empty
            continue

        # Keep non-time-indexed data in both halves so metadata is preserved.
        before_payload[key] = value
        after_payload[key] = value

    return (
        before_payload if before_has_rows else None,
        after_payload if after_has_rows else None,
    )


def main() -> None:
    # open TRBD001_all_stim_params.parq
    stim_params_df = pd.read_parquet("TRBD001_all_stim_params.parq")
    # sort stim_params_df by "CT_times" in ascending order
    stim_params_df = stim_params_df.sort_values(by="CT_times", ascending=True)
    # get the first "CT_times" when right_suspend_amplitude is not 0 and not NaN
    first_nonzero_ct_times = stim_params_df.query("right_suspend_amplitude != 0 and right_suspend_amplitude.notna()")["CT_times"].iloc[0]
    print(f"First non-zero right_suspend_amplitude CT_times: {first_nonzero_ct_times}")
    # 2025-06-04 22:56:14-05:00
    pkl_paths = sorted(BASE_DIR.glob(FILE_GLOB))
    if not pkl_paths:
        print(f"No files found with pattern {FILE_GLOB} under {BASE_DIR}")
        return

    all_records = [parse_file_record(path) for path in pkl_paths]

    by_date_side: dict[tuple[pd.Timestamp, str], list[FileRecord]] = defaultdict(list)
    for rec in all_records:
        by_date_side[(rec.date_start, rec.side)].append(rec)

    merged_records: list[MergedRecord] = []
    for _, candidates in sorted(by_date_side.items(), key=lambda item: (item[0][1], item[0][0])):
        merged_records.append(merge_candidates_for_day(candidates))

    by_side: dict[str, list[MergedRecord]] = defaultdict(list)
    for rec in merged_records:
        by_side[rec.side].append(rec)

    total_written = 0
    for side, side_records in sorted(by_side.items()):
        side_records = sorted(side_records, key=lambda r: r.date)
        groups = split_by_gap(side_records, MAX_GAP_DAYS)

        for group_idx, group in enumerate(groups, start=1):
            payloads = [rec.payload for rec in group]
            merged_payload = concat_dict_payload(payloads)

            merged_payload["merged_source_channels"] = [
                channel
                for rec in group
                for channel in rec.source_channels
            ]
            merged_payload["merged_source_paths"] = [
                path
                for rec in group
                for path in rec.source_paths
            ]

            should_split_visit = group[0].date == VISIT_SPLIT_START_DATE

            if should_split_visit:
                before_payload, after_payload = split_payload_by_timestamp(
                    merged_payload,
                    VISIT_SPLIT_TIMESTAMP,
                )

                if before_payload is not None:
                    before_name = build_output_name_with_suffix(
                        side,
                        before_payload,
                        group_idx,
                        "before_split"
                    )
                    before_path = OUTPUT_DIR / before_name
                    save_pickle(before_path, before_payload)
                    total_written += 1
                    print(
                        f"Wrote {before_path} from {len(group)} files "
                        f"(<= {VISIT_SPLIT_TIMESTAMP})"
                    )

                if after_payload is not None:
                    after_name = build_output_name_with_suffix(
                        side,
                        after_payload,
                        group_idx,
                        "after_split"
                    )
                    after_path = OUTPUT_DIR / after_name
                    save_pickle(after_path, after_payload)
                    total_written += 1
                    print(
                        f"Wrote {after_path} from {len(group)} files "
                        f"(> {VISIT_SPLIT_TIMESTAMP})"
                    )

                if before_payload is None and after_payload is None:
                    print(
                        f"Skipped split output for visit {group_idx:02d} {side}: "
                        "no time-indexed rows found around split timestamp"
                    )
                continue

            out_name = build_output_name(side, merged_payload, group_idx)
            out_path = OUTPUT_DIR / out_name
            save_pickle(out_path, merged_payload)
            total_written += 1

            print(
                f"Wrote {out_path} from {len(group)} files "
                f"({group[0].date.date()} -> {group[-1].date.date()})"
            )

    print(f"Done. Wrote {total_written} concatenated single-channel visit files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
