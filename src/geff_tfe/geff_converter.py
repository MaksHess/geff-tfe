import os
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import rustworkx as rx
from colorizer_data.types import FeatureInfo, FeatureMetadata
from colorizer_data.utils import (
    cast_feature_to_info_type,
    sanitize_key_name,
    update_metadata,
    write_data_array,
)

from geff_tfe import track_io as tio
from geff_tfe.types_v2 import (
    Backdrop3dMetadata,
    ColorizerMetadata,
    DatasetManifest,
    Frames3dMetadata,
    Segmentation3dMetadata,
    TracksMetadata,
)


def _write_feature(
    data: np.ndarray, name: str, outpath: Path, index: int
) -> FeatureMetadata:
    info = FeatureInfo(label=name, key=name, column_name=name)
    data, info = cast_feature_to_info_type(data, info)

    finite = data[np.isfinite(data)]
    fmin = float(np.nanmin(finite)) if finite.size else 0.0
    fmax = float(np.nanmax(finite)) if finite.size else 0.0

    if data.dtype == np.float64:
        data = data.astype(np.float32)

    filename = write_data_array(data, outpath, f"feature_{index}")

    metadata: FeatureMetadata = {
        "name": info.label,
        "data": filename,
        "unit": info.unit,
        "type": info.type,
        "key": info.key,
        "min": fmin,
        "max": fmax,
        "description": info.description,
        "categories": info.categories or [],
    }
    return metadata


def _infer_total_frames(source: Path):
    import bioio_ome_zarr
    from bioio import BioImage

    img = BioImage(source, reader=bioio_ome_zarr.Reader)
    return img.dims.T


def convert_geff_to_tfe(
    input_geff_dir: Path | str,
    output_tfe_dir: Path | str,
    *,
    feature_column_names: list[str] | None = None,
    segmentation_id_column: str | None = None,
    segmentation: Segmentation3dMetadata | None = None,
    backdrops: list[Backdrop3dMetadata] | None = None,
    metadata: ColorizerMetadata | None = None,
):
    """
    Converts a subset of GEFF to TFE-format (>=2.0.0).

    Concept mapping GEFF -> TFE:
    node_ids -> not used by TFE*, maintained for round trip conversion
    node_props -> features
    edge_ids -> not used by TFE*, maintained for round trip conversion
    edge_props -> not used by TFE, maintained for round trip conversion
    axes[type=="time"] -> times_column; warning if unit != "frame"
    axes[type=="space"] -> centroid_[zyx]_column; exception if not 3d.
    track_node_props[tracklet] -> track_column; if missing, inferred from tracking graph.
    track_node_props[lineage] -> features[lineage_id]; if missing, inferred from tracking graph.
    related_objects[type=="labels"] -> frames_3d[segmentation] + seg_id
    related_objects[type=="image"] -> frames_3d[backdrops]

    *used implicitly inside this function IF generating track and lineage ids

    Args:
        input_geff_dir: A GEFF directory to covert to TFE format.
        output_tfe_dir: The directory to write the dataset to. All dataset files
            will be placed directly within this directory.
        feature_column_names: An array of feature column names. If a value is
            provided, ONLY the provided column names will be parsed as features; otherwise, ALL
            columns that aren't specified as a backdrop or a data column (e.g. object ID, time,
            track, etc.) will be treated as a feature. Defaults to `None`.
        segmentation_id_column: The node property on the GEFF graph that represents the label id.
            If left empty the GEFF requires a related object of type "labels" (node_prop->segmentation_id_column)
        segmentation: A `Segmentation3dMetadata` object pointing to associated segmentations. If not
            provided use the first GEFF related object of type "labels".
        backdrops: A list of `Backdrop3dMetadata` objects pointing to backdrop images to be displayed. If not
            provided attempts to infer them from GEFF `related_objects` of type "image".
        metadata: Metadata to include in the dataset's manifest, such
            as the dataset name, author, dataset description, frame resolution, and time units.
            See `ColorizerMetadata` for more information. Note that some information will be
            written automatically, such as a timestamp and revision number.

    Example:
        ```python
            from colorizer_data import convert_geff_to_tfe

            # 1. Assuming "fully compatible" GEFF, including `related_objects` pointing to a segmentation and specifying label_id (seg_id) column.
            convert_geff_to_tfe("dataset.ome.zarr/tracks/my_tracks.geff", "dataset.ome.zarr/tracks/my_tracks.tfe")
            # 2. Manually specifying segmentation and backdrop images
            seg_id_node_prop = "label_id"
            segmentation = Segmentation3dMetadata(
                name="nucleus",
                key=sanitize_key_name("nucleus"),
                source="path/to/image.ome.zarr/labels/nucleus",
                channel_index=0,
            )
            backdrops=[
                Backdrop3dMetadata(
                    name="H2B",
                    key=sanitize_key_name("H2B"),
                    source="path/to/image.ome.zarr",
                    channel_index=1,
                ),
                Backdrop3dMetadata(
                    name="mem9",
                    key=sanitize_key_name("mem9"),
                    source="path/to/image.ome.zarr",
                    channel_index=0,
                )
            ]
            convert_geff_to_tfe(
                "dataset.ome.zarr/tracks/my_tracks.geff",
                "dataset.ome.zarr/tracks/my_tracks.tfe",
                segmentation_id_column=seg_id_node_prop,
                segmentations=segmentations,
                backdrops=backdrops,
            )
        ```
    """
    original_cwd = Path.cwd()
    output_tfe_dir = Path(output_tfe_dir)
    output_tfe_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(output_tfe_dir)

        rx_graph, geff_metadata = tio.read_geff(input_geff_dir, return_metadata=True)

        axes = geff_metadata.axes or []
        time_axes = [axis for axis in axes if axis.type == "time"]
        if len(time_axes) != 1:
            raise ValueError(f"Expected exactly one time axis, found {len(time_axes)}.")
        time_axis = time_axes[0]
        if time_axis.unit != "frame":
            warnings.warn(
                f"Time axis {time_axis.name!r} has unit {time_axis.unit!r}, expected 'frame'.",
                stacklevel=2,
            )

        space_axes = [axis for axis in axes if axis.type == "space"]
        if len(space_axes) != 3:
            raise ValueError(
                "convert_geff_to_tfe only supports 3d GEFF datasets, found "
                f"{len(space_axes)} space axes."
            )
        z_col, y_col, x_col = (axis.name for axis in space_axes)

        # Prefer GEFF-provided tracklet/lineage ids, otherwise infer from graph topology.
        track_node_props = geff_metadata.track_node_props or {}
        track_col = track_node_props.get("tracklet")
        if track_col is None:
            tio.add_track_id(rx_graph)
            track_col = tio.TRACK_ID_COLUMN

        lineage_col = track_node_props.get("lineage")
        if lineage_col is None:
            tio.add_lineage_id(rx_graph)
            lineage_col = tio.LINEAGE_ID_COLUMN

        tio.add_edge_type(rx_graph, overwrite=True)

        if segmentation_id_column is None:
            for related_object in geff_metadata.related_objects or []:
                if related_object.type == "labels":
                    labels_object = related_object
                    segmentation_id_column = (
                        labels_object.node_prop or labels_object.label_prop
                    )
                    if segmentation_id_column is None:
                        raise ValueError(
                            "Found a related object of type 'labels', but it's missing `node_prop` "
                            "(or deprecated `label_prop`)."
                        )
                    if segmentation is None:
                        segmentation_name = Path(labels_object.path).name
                        segmentation = Segmentation3dMetadata(
                            name=segmentation_name,
                            key=sanitize_key_name(segmentation_name),
                            source=labels_object.path,
                            channel_index=0,
                        )
                    break
        if segmentation is None:
            raise ValueError(
                "No related object of type 'labels' found. Provide `segmentation` and `segmentation_id_column`."
            )

        if backdrops is None:
            backdrops = []

        total_frames = _infer_total_frames(Path(segmentation.source))
        frames_3d = Frames3dMetadata(
            segmentations=[segmentation],
            backdrops=backdrops,
            total_frames=total_frames,
        )

        df_nodes, df_edges = tio.rx_to_geff_like(rx_graph)
        if segmentation_id_column not in df_nodes.columns:
            raise ValueError(
                f"Segmentation ID column {segmentation_id_column!r} (from related object "
                f"{labels_object.path!r}) not found in GEFF node properties."
            )
        if lineage_col not in df_nodes.columns:
            raise ValueError(
                f"Lineage ID column {lineage_col!r} not found in GEFF node properties."
            )

        times = df_nodes[time_axis.name].to_numpy()
        seg_ids = df_nodes[segmentation_id_column].to_numpy()

        reserved_columns = {
            tio.NODE_ID_COLUMN,
            time_axis.name,
            z_col,
            y_col,
            x_col,
            track_col,
            segmentation_id_column,
        }
        if feature_column_names is not None:
            missing = set(feature_column_names) - set(df_nodes.columns)
            if missing:
                raise ValueError(
                    f"feature_column_names not found in GEFF node properties: {missing}"
                )
            reserved = set(feature_column_names) & reserved_columns
            if reserved:
                raise ValueError(
                    f"feature_column_names must not include reserved columns: {reserved}"
                )
            feature_columns = feature_column_names
        else:
            feature_columns = [c for c in df_nodes.columns if c not in reserved_columns]

        features = [
            _write_feature(df_nodes[column].to_numpy(), column, output_tfe_dir, i)
            for i, column in enumerate(feature_columns)
        ]

        times_filename = write_data_array(times, output_tfe_dir, "times")
        seg_ids_filename = write_data_array(seg_ids, output_tfe_dir, "seg_ids")

        centroids_stacked = np.ravel(df_nodes.select(x_col, y_col, z_col))
        centroids_filename = write_data_array(centroids_stacked, output_tfe_dir, 'centroids')

        # centroids_filename = "centroids.parquet"
        # pl.DataFrame(
        #     {
        #         "centroid_z": df_nodes[z_col],
        #         "centroid_y": df_nodes[y_col],
        #         "centroid_x": df_nodes[x_col],
        #     }
        # ).write_parquet(output_tfe_dir / centroids_filename)

        # Round trip files: the full object-level GEFF node/edge tables (including generated
        # track/lineage ids), keyed by the original GEFF node ids.
        tio._write_geff_like(output_tfe_dir, (df_nodes, df_edges))

        # Tracklet-level graph: one node per track id, edges denote SPLIT/MERGE relationships
        # between tracks. Used for lineage-tree layout (see `generation`).
        track_ids_sorted = sorted(set(df_nodes[track_col].to_list()))
        id_to_rx = {track_id: i for i, track_id in enumerate(track_ids_sorted)}

        track_graph = rx.PyDiGraph()
        track_graph.attrs = {tio.PERSISTENT_NODE_ID_ATTR: id_to_rx}
        track_graph.add_nodes_from([{} for _ in track_ids_sorted])

        for u, v in rx_graph.edge_list():
            edge_data = rx_graph.get_edge_data(u, v)
            if edge_data[tio.EDGE_TYPE_COLUMN] == tio.EdgeType.TRACK:
                continue
            parent_idx = id_to_rx[rx_graph[u][track_col]]
            child_idx = id_to_rx[rx_graph[v][track_col]]
            if not track_graph.has_edge(parent_idx, child_idx):
                track_graph.add_edge(parent_idx, child_idx, {})

        tio.add_topological_generation(track_graph, overwrite=True)

        df_track_nodes, df_track_edges = tio.rx_to_geff_like(
            track_graph,
            node_id_column=tio.TRACK_NODE_ID_COLUMN,
            edge_id_column_start=tio.TRACK_EDGE_ID_COLUMN_START,
            edge_id_column_end=tio.TRACK_EDGE_ID_COLUMN_END,
        )
        df_track_nodes.write_parquet(output_tfe_dir / tio.TRACK_NODES_FILE_NAME)
        df_track_edges.write_parquet(output_tfe_dir / tio.TRACK_EDGES_FILE_NAME)

        tracks_metadata = TracksMetadata(
            track_ids=tio.TRACK_NODES_FILE_NAME,
            track_edges=tio.TRACK_EDGES_FILE_NAME,
            node_ids=tio.NODES_FILE_NAME,
            node_edges=tio.EDGES_FILE_NAME,
        )

        if metadata is None:
            metadata = ColorizerMetadata()
        update_metadata(metadata, default_name=output_tfe_dir.name)

        manifest = DatasetManifest(
            features=features,
            tracks=[tracks_metadata],
            centroids=centroids_filename,
            times=times_filename,
            seg_ids=seg_ids_filename,
            metadata=metadata,
            frames3d=frames_3d,
        )
        manifest.write(output_tfe_dir)

    finally:
        os.chdir(original_cwd)
