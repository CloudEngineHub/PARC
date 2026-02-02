import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

"""
Example script for reading the motion data format
"""

import parc.util.file_io as file_io

MOTION_FILE = "data/motion_terrains/sfu.pkl"

ms_file_data = file_io.load_ms_file(MOTION_FILE)

print(ms_file_data.motion_data.root_pos)
print(ms_file_data.motion_data.root_rot)
print(ms_file_data.motion_data.joint_rot)
print(ms_file_data.motion_data.body_contacts)
print(ms_file_data.motion_data.fps)