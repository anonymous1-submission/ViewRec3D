import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def _process_image_folder(image_folder: Path):
    files = [p for p in image_folder.iterdir() if p.is_file()]
    file_map = {p.name: p for p in files}

    if "original.jpeg" not in file_map:
        return None
    res_list = []
    for p in files:
        if p.name.startswith("generated") and p.suffix == ".jpeg":
            json_name = f"{p.stem}.json"
            if json_name not in file_map:
                continue
            res_list.append({
                "original_path": str(file_map["original.jpeg"]),
                "generated_path": str(file_map[p.name]),
                "json_path": str(file_map[json_name]),
            })

    return res_list


def build_samples(
    dataset_root: Path,
    data_ratio: float,
    data_angle: int,
    num_workers: int = 8,
):
    cache_path = dataset_root / f"indexing_{data_angle}.json"
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data_index = json.load(f)
    else:
        image_folders = [p for p in dataset_root.iterdir() if p.is_dir()]

        data_index = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_process_image_folder, image_folders)
            for item in tqdm(results, total=len(image_folders), desc="Indexing folders"):
                if item is not None:
                    data_index.extend(item)

        if cache_path is not None:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data_index, f, ensure_ascii=False)

    data_size = len(data_index)
    data_index = data_index[:int(data_size*data_ratio)]

    return data_index





