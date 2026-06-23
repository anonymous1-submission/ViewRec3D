import json
from pathlib import Path


def load_annotation(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = data["target"]
    change_orientation = data["change_orientation"]
    pose = data["pose"]
    intrinsic = data["K"]
    scale = data["pose"][7]
    intrinsic[0][0] = intrinsic[0][0] / scale
    intrinsic[1][1] = intrinsic[1][1] / scale
    return change_orientation, pose, intrinsic

