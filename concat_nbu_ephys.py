import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


FILE_PREFIX = "spectra_fooof_bandpower_"
FILE_GLOB = "*/rawlfp/raw/spectra_fooof_bandpower_*.pkl"
BASE_DIR = Path("/mnt/labworlds/Provenza/NBUPipeline/TRBD001")
OUTPUT_DIR = Path("nbu_concat")
MAX_GAP_DAYS = 4


@dataclass
class FileRecord:
    path: Path
    date: pd.Timestamp
    channel: str


def parse_file_record(path: Path) -> FileRecord:
    # Expected date location: <base>/<YYYY-MM-DD>/rawlfp/raw/<file>.pkl
    date_str = path.parent.parent.parent.name
    date = pd.Timestamp(date_str)

    stem = path.stem
    if not stem.startswith(FILE_PREFIX):
        raise ValueError(f"Unexpected filename format: {path.name}")
    channel = stem[len(FILE_PREFIX):]

    return FileRecord(path=path, date=date, channel=channel)


def split_by_gap(records: list[FileRecord], max_gap_days: int) -> list[list[FileRecord]]:
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


def build_output_name(channel: str, group: list[FileRecord], group_idx: int) -> str:
    start = group[0].date.strftime("%Y-%m-%d")
    end = group[-1].date.strftime("%Y-%m-%d")
    return f"{FILE_PREFIX}{channel}_visit{group_idx:02d}_{start}_to_{end}.pkl"


def main() -> None:
    pkl_paths = sorted(BASE_DIR.glob(FILE_GLOB))
    if not pkl_paths:
        print(f"No files found with pattern {FILE_GLOB} under {BASE_DIR}")
        return

    records = [parse_file_record(path) for path in pkl_paths]
    by_channel: dict[str, list[FileRecord]] = defaultdict(list)
    for rec in records:
        by_channel[rec.channel].append(rec)

    total_written = 0
    for channel, channel_records in sorted(by_channel.items()):
        channel_records = sorted(channel_records, key=lambda r: r.date)
        groups = split_by_gap(channel_records, MAX_GAP_DAYS)

        for group_idx, group in enumerate(groups, start=1):
            payloads = [load_pickle(rec.path) for rec in group]
            merged_payload = concat_dict_payload(payloads)

            out_name = build_output_name(channel, group, group_idx)
            out_path = OUTPUT_DIR / out_name
            save_pickle(out_path, merged_payload)
            total_written += 1

            print(
                f"Wrote {out_path} from {len(group)} files "
                f"({group[0].date.date()} -> {group[-1].date.date()})"
            )

    print(f"Done. Wrote {total_written} concatenated visit files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
