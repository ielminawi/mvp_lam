"""Inspect paired Cable Routing views; no LAM inference or offset fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import tensorflow_datasets as tfds


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-name", default="berkeley_cable_routing-train.tfrecord-00000-of-00064")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    if Path(args.shard_name).name != args.shard_name:
        raise ValueError("Shard name must be a filename")
    shard = args.data_dir / args.shard_name
    builder = tfds.builder_from_directory(str(args.data_dir))
    args.output_dir.mkdir(parents=True)
    cameras = ("image", "top_image")
    overview_rows = []
    report = {
        "scope": __doc__, "shard": str(shard),
        "input_sha256": {path.name: digest(path) for path in
                         (shard, args.data_dir / "dataset_info.json", args.data_dir / "features.json")},
        "limitations": "Step indices are candidate alignment, not timestamp verification. Image statistics do not prove static cameras. No camera arrangement grouping or source deduplication is established.",
        "episodes": [],
    }
    for record_index, record in enumerate(tf.data.TFRecordDataset(str(shard))):
        episode = builder.info.features.deserialize_example(record)
        steps = episode["steps"]
        if isinstance(steps, dict):
            steps = tf.data.Dataset.from_tensor_slices(steps)
        images = {camera: [] for camera in cameras}
        instruction = ""
        flags = {name: [] for name in ("is_first", "is_last", "is_terminal")}
        for step in steps:
            for camera in cameras:
                images[camera].append(step["observation"][camera].numpy())
            if not instruction:
                instruction = step["observation"]["natural_language_instruction"].numpy().decode("utf-8")
            for name in flags:
                flags[name].append(bool(step[name].numpy()))
        entry = {"record_index": record_index, "steps": len(images["image"]),
                 "instruction": instruction, "cameras": {},
                 "flag_indices": {name: np.flatnonzero(value).tolist() for name, value in flags.items()}}
        for camera in cameras:
            if not images[camera]:
                entry["cameras"][camera] = {"empty": True}
                continue
            frames = np.stack(images[camera])
            images[camera] = frames
            flat = frames.reshape(len(frames), -1)
            entry["cameras"][camera] = {
                "shape": list(frames.shape), "dtype": str(frames.dtype),
                "all_black_frame_indices": np.flatnonzero(np.max(flat, axis=1) == 0).tolist(),
                "constant_frame_indices": np.flatnonzero(np.std(flat, axis=1) == 0).tolist(),
                "adjacent_mean_absolute_pixel_change": float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean()) if len(frames) > 1 else None,
            }
        if entry["steps"]:
            row_image = Image.new("RGB", (768, 216), "white")
            row_draw = ImageDraw.Draw(row_image)
            for column, (camera, index) in enumerate(
                [(camera, index) for index in (0, entry["steps"] - 1) for camera in cameras]
            ):
                x = column * 192
                row_draw.text((x + 3, 5), f"ep {record_index} {camera} t={index}", fill="black")
                row_image.paste(Image.fromarray(images[camera][index]).resize((192, 192)), (x, 24))
            overview_rows.append(row_image)
        if record_index < 3 and entry["steps"]:
            indices = np.unique(np.linspace(0, entry["steps"] - 1, min(6, entry["steps"]), dtype=int))
            sheet = Image.new("RGB", (512, 280 * len(indices)), "white")
            draw = ImageDraw.Draw(sheet)
            for row, index in enumerate(indices):
                for column, camera in enumerate(cameras):
                    x, y = column * 256, row * 280
                    draw.text((x + 5, y + 5), f"{camera} | step {index}", fill="black")
                    sheet.paste(Image.fromarray(images[camera][index]).resize((256, 256)), (x, y + 24))
            name = f"episode_{record_index:05d}_paired_views.png"
            sheet.save(args.output_dir / name)
            entry["contact_sheet"] = name
        report["episodes"].append(entry)
        print(f"Record {record_index}: {entry['steps']} steps; instruction={instruction!r}", flush=True)
        for camera, info in entry["cameras"].items():
            print(f"  {camera}: {info}", flush=True)
    report["episode_count"] = len(report["episodes"])
    report["overview_sheets"] = []
    for start in range(0, len(overview_rows), 5):
        rows = overview_rows[start:start + 5]
        overview = Image.new("RGB", (768, 216 * len(rows)), "white")
        for row, image in enumerate(rows):
            overview.paste(image, (0, row * 216))
        name = f"all_episodes_overview_{start // 5:02d}.png"
        overview.save(args.output_dir / name)
        report["overview_sheets"].append(name)
    (args.output_dir / "inspection.json").write_text(json.dumps(report, indent=2))
    print(f"\nInspected {report['episode_count']} episodes. Saved {args.output_dir}")
    print("Review contact sheets for camera stability and candidate temporal alignment. No LAM inference performed.")


if __name__ == "__main__":
    main()
