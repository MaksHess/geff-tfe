import json
import os
import shutil
from pathlib import Path

import lsdata

from geff_tfe.geff_converter import Segmentation3dMetadata, convert_geff_to_tfe
from geff_tfe.types_v2 import Backdrop3dMetadata


def main():
    dataset_dir = lsdata.download("001-full")
    deconv_dir = dataset_dir / "deconv.ome.zarr"
    input_geff_dir = deconv_dir / "tracks" / "nucleus.geff"
    output_tfe_dir = input_geff_dir.parent / "nucleus.tfe"
    symlink_dataset = "deconv.ome.zarr"

    fixed_geff_dir = _fix_geff_tracklet_id(input_geff_dir)

    output_tfe_dir.mkdir(exist_ok=True)
    if not (output_tfe_dir / symlink_dataset).exists():
        os.symlink(deconv_dir, output_tfe_dir / symlink_dataset)

    segmentation = Segmentation3dMetadata(
        name="nucleus",
        key="nucleus",
        source=(Path(symlink_dataset) / "labels" / "nucleus").as_posix(),
        channel_index=0,
    )
    backdrops = [
        Backdrop3dMetadata(
            name="H2B", key="h2b", source=symlink_dataset, channel_index=1
        ),
        Backdrop3dMetadata(
            name="mem9", key="mem9", source=symlink_dataset, channel_index=0
        ),
    ]

    convert_geff_to_tfe(
        fixed_geff_dir,
        output_tfe_dir,
        segmentation=segmentation,
        segmentation_id_column="label_id",
        backdrops=backdrops,
    )


# TODO: remove with dataset release that fixes this.
def _fix_geff_tracklet_id(geff_dir: Path) -> Path:
    """Fix the geff graph in v0.1 dataset to point to 'track_id' as the tracklet node property."""

    with open(geff_dir / "zarr.json") as f:
        data = json.load(f)

    track_node_props = data["attributes"]["geff"].get("track_node_props")
    if track_node_props is None:
        fixed_geff_dir = geff_dir.parent / f"{geff_dir.stem}-fixed{geff_dir.suffix}"
        shutil.copytree(geff_dir, fixed_geff_dir, dirs_exist_ok=True)
        data["attributes"]["geff"]["track_node_props"] = {"tracklet": "track_id"}
        with open(fixed_geff_dir / "zarr.json", "w") as f:
            json.dump(data, f)
    else:
        fixed_geff_dir = geff_dir
    return fixed_geff_dir


if __name__ == "__main__":
    main()
