from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataclasses_json import DataClassJsonMixin, LetterCase, config

from colorizer_data.types import ColorizerMetadata, FeatureMetadata, FeatureType


def _get_config() -> dict[str, Any]:
    return config(letter_case=LetterCase.CAMEL, undefined=None)["dataclasses_json"]


@dataclass
class TracksMetadata(DataClassJsonMixin):
    dataclass_json_config = _get_config()

    track_ids: str
    track_edges: str
    node_ids: str
    node_edges: str


@dataclass
class Backdrop2dMetadata(DataClassJsonMixin):
    dataclass_json_config = _get_config()

    name: str
    key: str
    frames: list[str]
    description: str


@dataclass
class Frames2dMetadata(DataClassJsonMixin):
    dataclass_json_config = _get_config()

    segmentations: list[Backdrop2dMetadata]
    backdrops: list[Backdrop2dMetadata]


@dataclass
class Segmentation3dMetadata(DataClassJsonMixin):
    dataclass_json_config = _get_config()

    name: str
    key: str
    source: str
    channel_index: int
    description: str = ""


@dataclass
class Backdrop3dMetadata(DataClassJsonMixin):
    dataclass_json_config = _get_config()

    name: str
    key: str
    source: str
    channel_index: int
    description: str = ""
    min: float | None = None
    max: float | None = None


@dataclass
class Frames3dMetadata(DataClassJsonMixin):
    dataclass_json_config = _get_config()

    segmentations: list[Segmentation3dMetadata]
    backdrops: list[Backdrop3dMetadata]
    total_frames: int


@dataclass
class DatasetManifest(DataClassJsonMixin):
    dataclass_json_config = _get_config()

    features: list[FeatureMetadata]
    tracks: list[TracksMetadata]
    centroids: str
    times: str
    seg_ids: str
    metadata: ColorizerMetadata
    bounds: str | None = None
    outliers: str | None = None
    frames2d: Frames2dMetadata | None = None
    frames3d: Frames3dMetadata | None = None

    def write(self, output_path: Path) -> None:
        with open(output_path / "manifest.json", mode='w') as f:
            f.writelines(self.to_json(indent=2))


def _get_features_tracks_and_metadata() -> tuple[
    list[FeatureMetadata], list[TracksMetadata], ColorizerMetadata
]:
    features = [
        FeatureMetadata(
            data="feature_0.parquet",
            name="test",
            key="test",
            unit="",
            type=FeatureType.CONTINUOUS,
            categories=[],
            min=0.0,
            max=10.0,
            description="",
        )
    ]
    tracks = [
        TracksMetadata(
            track_ids="track_ids.parquet",
            track_edges="track_edges.parquet",
            node_ids="node_ids.parquet",
            node_edges="node_deges.parquet",
        )
    ]
    metadata = ColorizerMetadata(
        name="001-full", description="", _writer_version="v1.8.0"
    )
    return features, tracks, metadata


def _get_dummy_manifest_3d() -> DatasetManifest:
    features, tracks, metadata = _get_features_tracks_and_metadata()
    frames3d = Frames3dMetadata(
        segmentations=[
            Segmentation3dMetadata(
                name="nucleus",
                key="nucleus",
                source="deconv.ome.zarr/labels/nucleus",
                channel_index=0,
                description="",
            )
        ],
        backdrops=[
            Backdrop3dMetadata(
                name="H2B",
                key="h2b",
                source="deconv.ome.zarr",
                channel_index=1,
                description="",
                min=0,
                max=8000,
            ),
            Backdrop3dMetadata(
                name="mem9",
                key="mem9",
                source="deconv.ome.zarr",
                channel_index=0,
                description="",
                min=0,
                max=27000,
            ),
        ],
        total_frames=10,
    )

    return DatasetManifest(
        features=features,
        tracks=tracks,
        centroids="centroids.parquet",
        times="times.parquet",
        seg_ids="seg_ids.parquet",
        metadata=metadata,
        frames3d=frames3d,
    )


def _get_dummy_manifest_2d() -> DatasetManifest:
    n_frames = 10
    features, tracks, metadata = _get_features_tracks_and_metadata()
    frames2d = Frames2dMetadata(
        segmentations=[
            Backdrop2dMetadata(
                name="nucleus",
                key="nucleus",
                frames=[f"frames/img_{i}.png" for i in range(n_frames)],
                description="",
            )
        ],
        backdrops=[
            Backdrop2dMetadata(
                name="H2B",
                key="h2b",
                frames=[f"backdrop_H2B/img_{i}.png" for i in range(n_frames)],
                description="",
            ),
            Backdrop2dMetadata(
                name="mem9",
                key="mem9",
                frames=[f"backdrop_mem9/img_{i}.png" for i in range(n_frames)],
                description="",
            )
        ],
    )
    return DatasetManifest(
        features=features,
        tracks=tracks,
        centroids="centroids.parquet",
        times="times.parquet",
        seg_ids="seg_ids.parquet",
        metadata=metadata,
        frames2d=frames2d,    
    )
