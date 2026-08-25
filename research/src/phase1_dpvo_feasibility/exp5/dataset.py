"""In-memory reuse of the fixed Exp4 EuRoC frame identity protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..exp4.dataset import _camera_rows, discover_sequences


@dataclass(frozen=True)
class FrameRecord:
    dataset: str
    dataset_group: str
    split: str
    sequence: str
    frame_id: int
    stream_index: int
    timestamp: int
    image_path: str

    def __post_init__(self) -> None:
        if self.dataset != "euroc":
            raise ValueError(f"unsupported dataset: {self.dataset}")
        if self.frame_id < 0 or self.stream_index < 0 or self.timestamp <= 0:
            raise ValueError("frame identity values must be non-negative and timestamp positive")
        image = Path(self.image_path)
        if not image.is_absolute() or image.stem != str(self.timestamp):
            raise ValueError("image path must be absolute and its stem must equal timestamp")

    @property
    def identity(self) -> tuple[int, int]:
        return self.frame_id, self.timestamp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameRecord":
        expected = set(cls.__dataclass_fields__)
        missing, extra = expected - set(payload), set(payload) - expected
        if missing or extra:
            raise ValueError(f"frame schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        values = dict(payload)
        for key in ("frame_id", "stream_index", "timestamp"):
            values[key] = int(values[key])
        return cls(**values)


def configured_sequences(config: dict[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for split in ("train", "val", "test"):
        sequences = config["dataset"]["splits"].get(split, [])
        if len(sequences) != 1:
            raise ValueError(f"Exp5-0 requires exactly one {split} sequence")
        result.append(str(sequences[0]))
    if len(set(result)) != len(result):
        raise ValueError("Exp5-0 sequences must be unique across splits")
    return tuple(result)


def load_sequence_records(
    config: dict[str, Any], sequence: str, *, limit: int | None = None,
) -> list[FrameRecord]:
    """Select frames without copying images or writing a dataset index."""
    split_by_sequence = {
        str(item): split
        for split in ("train", "val", "test")
        for item in config["dataset"]["splits"][split]
    }
    if sequence not in split_by_sequence:
        raise ValueError(f"sequence is outside the fixed Exp5-0 protocol: {sequence}")
    discovered = discover_sequences(config)
    if sequence not in discovered:
        raise FileNotFoundError(f"configured EuRoC sequence was not discovered: {sequence}")
    item = discovered[sequence]
    camera = item.sequence_root / "mav0" / str(config["dataset"]["camera"])
    rows = _camera_rows(camera / "data.csv", camera / "data")
    stride, skip = int(config["dataset"]["stride"]), int(config["dataset"]["skip"])
    if stride <= 0 or skip < 0:
        raise ValueError("stride must be positive and skip non-negative")
    selected = list(enumerate(rows))[skip::stride]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    records = [
        FrameRecord(
            dataset=str(config["dataset"]["dataset_type"]),
            dataset_group=item.dataset_group,
            split=split_by_sequence[sequence],
            sequence=sequence,
            frame_id=int(frame_id),
            stream_index=int(stream_index),
            timestamp=int(timestamp),
            image_path=str((camera / "data" / filename).resolve()),
        )
        for stream_index, (frame_id, (timestamp, filename)) in enumerate(selected)
    ]
    if not records:
        raise RuntimeError(f"no frames selected for {sequence}")
    if [record.stream_index for record in records] != list(range(len(records))):
        raise AssertionError("stream indices must be contiguous")
    return records


def validate_real_images(records: list[FrameRecord]) -> list[list[int]]:
    """Decode real smoke images without importing either model runtime."""
    import cv2

    shapes: list[list[int]] = []
    for record in records:
        path = Path(record.image_path)
        if not path.is_file() or path.stem != str(record.timestamp):
            raise FileNotFoundError(f"invalid EuRoC frame path: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"OpenCV could not decode a color image: {path}")
        shapes.append([int(value) for value in image.shape])
    return shapes
