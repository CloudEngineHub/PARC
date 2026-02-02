import os
import sys
from pathlib import Path

import torch



# Ensure the repository root is on the import path so "parc" resolves consistently
# across platforms (editable installs on Windows sometimes fail to place the repo
# ahead of site-packages when invoking this script directly).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import parc.util.create_dataset as create_dataset
import parc.util.path_loader as path_loader
folder_path = [path_loader.resolve_path(Path("$DATA_DIR/releases_parc/dec_release/rerun_april272025_iter_2_boxes/"))]

create_dataset.create_dataset_yaml(
    folder_paths=folder_path,
    save_path=Path("output/datasets/rerun_april272025_iter_2_boxes.yaml"),
    char_filepath="data/assets/humanoid.xml",
    compute_preprocessing_data=False,
    override_old_hf_mask_inds=False,
    cut_some_classes_in_half=False,
    motion_classes_to_cut_in_half=[],
    max_terrain_dim_x=64,
    max_terrain_dim_y=64,
    min_num_frames=15
)