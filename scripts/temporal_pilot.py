"""Cache DINO features, check compatibility, and fit matched whole-window pilots."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms.functional import resize

from latent_action_model.genie.modules.temporal_pilot import WholeWindowLAM


RELEASE_HASH = "d388a242d603807eb97f6d80063d6d7daf000a55cfdf59498666eee87a2fe3b6"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision(directory):
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def provenance(args):
    repo = Path(__file__).resolve().parents[1]
    return {"git_commit": revision(repo), "git_diff": subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD"], text=True),
        "torch": torch.__version__, "python": sys.version,
        "gpu": torch.cuda.get_device_name(0), "checkpoint_sha256": sha256(args.checkpoint),
        "dino_commit": revision(args.dino_dir)}


def load_model(args):
    if sha256(args.checkpoint) != RELEASE_HASH:
        raise ValueError("Expected the verified released checkpoint")
    original_load = torch.hub.load

    def local_load(repo, name, *positional, **kwargs):
        if (repo, name) != ("facebookresearch/dinov2", "dinov2_vitb14_reg"):
            raise ValueError("Unexpected hub request")
        return original_load(str(args.dino_dir.resolve()), name, source="local", pretrained=False)

    with patch("torch.hub.load", side_effect=local_load):
        model = WholeWindowLAM(in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
            patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    if not all(key.startswith("lam.") for key in state):
        raise ValueError("Unexpected checkpoint prefix")
    model.load_state_dict({key.removeprefix("lam."): value for key, value in state.items()}, strict=True)
    return model.cuda().eval()


def read_config(path):
    config = json.loads(path.read_text())
    if config["horizon"] != 9 or config["offsets"] != {"two_frame": [0, 9], "four_frame": [0, 3, 6, 9]}:
        raise ValueError("This pilot uses fixed H=9 and offsets 0/3/6/9")
    for key in ("updates", "micro_batch_size", "accumulation_steps", "validation_every"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["precision"] != "float32":
        raise ValueError("Pilot currently supports float32 only")
    return config


def cache_features(args):
    manifest = json.loads((args.sample_dir / "manifest.json").read_text())
    if manifest["horizon_steps"] != 9:
        raise ValueError("Need the H=9 V3 preparation")
    sources = {s: {e["source_path"] for e in manifest["episodes"] if e["split"] == s}
               for s in ("train", "validation", "test")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sources[a] & sources[b]:
            raise ValueError("Source episode leakage")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = load_model(args)
    entries, counts = [], {"train": 0, "validation": 0}
    for entry in manifest["episodes"]:
        split = entry["split"]
        if split not in counts or (args.limit_episodes and counts[split] >= args.limit_episodes):
            continue
        with np.load(args.sample_dir / entry["file"], allow_pickle=False) as episode:
            frames, starts, ends = episode["frames"], episode["start_indices"], episode["end_indices"]
            np.testing.assert_array_equal(ends - starts, np.full(len(starts), 9))
            window_indices = starts[:, None] + np.array([0, 3, 6, 9])[None, :]
            # Same filtering for both models, including intermediate observations.
            valid = np.array([all(frames[i].std() > 0 for i in row) for row in window_indices])
            windows = window_indices[valid]
            if not len(windows):
                raise ValueError(f"No valid four-frame windows: {entry['file']}")
            required = np.unique(windows)
            filename = Path(entry["file"]).stem
            cache_path = args.output_dir / f"{filename}_patches.npy"
            patches = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32,
                                               shape=(len(required), 256, 768))
            for first in range(0, len(required), args.batch_size):
                last = min(first + args.batch_size, len(required))
                pixels = torch.from_numpy(frames[required[first:last]]).permute(0, 3, 1, 2).float() / 255.
                pixels = resize(pixels, [224, 224], antialias=True).cuda()
                with torch.no_grad():
                    features = model.dino_encoder.forward_features(model.dino_transform(pixels))["x_norm_patchtokens"]
                if not torch.isfinite(features).all():
                    raise ValueError("Non-finite features")
                patches[first:last] = features.cpu().numpy()
            patches.flush()
            del patches
            metadata_path = args.output_dir / f"{filename}_windows.npz"
            np.savez(metadata_path, windows=np.searchsorted(required, windows),
                original_frame_indices=windows, targets_raw=episode["targets_raw"][valid])
        counts[split] += 1
        entries.append({**entry, "patch_file": cache_path.name, "window_file": metadata_path.name,
                        "windows": len(windows), "filtered_intermediate_windows": int((~valid).sum()),
                        "patch_sha256": sha256(cache_path), "window_sha256": sha256(metadata_path)})
        print(f"Cached {split}: {filename}, {len(windows)} windows", flush=True)
    if not all(counts.values()):
        raise ValueError("Both train and validation episodes required")
    report = {"status": "complete", "scope": "single-view pilot; test not cached",
        "provenance": provenance(args), "sample_manifest_sha256": sha256(args.sample_dir / "manifest.json"),
        "offsets": [0, 3, 6, 9], "limit_episodes_per_split": args.limit_episodes,
        "preprocessing": "uint8/255; torchvision bilinear antialias resize224; model normalization; float32",
        "episodes": entries}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    print("Cache complete:", counts, "No LAM training or test access.")


class Windows:
    def __init__(self, directory, manifest, split):
        self.directory, self.entries, self.references = directory, [], []
        self.open_files = OrderedDict()
        for entry in manifest["episodes"]:
            if entry["split"] != split:
                continue
            with np.load(directory / entry["window_file"], allow_pickle=False) as saved:
                windows = saved["windows"]
            index = len(self.entries)
            self.entries.append((entry, windows))
            self.references.extend((index, row) for row in range(len(windows)))
        if not self.references:
            raise ValueError(f"Empty {split}")

    def __len__(self):
        return len(self.references)

    def batch(self, indices, variant):
        result = []
        for i in indices:
            entry_index, row = self.references[int(i)]
            entry, windows = self.entries[entry_index]
            key = entry["patch_file"]
            if key not in self.open_files:
                self.open_files[key] = np.load(self.directory / key, mmap_mode="r", allow_pickle=False)
                if len(self.open_files) > 8:
                    self.open_files.popitem(last=False)
            self.open_files.move_to_end(key)
            selected = windows[row] if variant == "four_frame" else windows[row][[0, 3]]
            result.append(np.array(self.open_files[key][selected], copy=True))
        return torch.from_numpy(np.stack(result)).cuda()


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate(model, dataset, variant, batch_size):
    total, copy_total, n = 0., 0., 0
    histogram = np.zeros(16, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for first in range(0, len(dataset), batch_size):
            patches = dataset.batch(range(first, min(first + batch_size, len(dataset))), variant)
            output = model.reconstruct_window(patches)
            error = ((output["recon"] - output["target"]) ** 2).mean(dim=(1, 2, 3))
            copy_error = ((patches[:, :1] - patches[:, -1:]) ** 2).mean(dim=(1, 2, 3))
            total += error.double().sum().item()
            copy_total += copy_error.double().sum().item()
            histogram += np.bincount(output["indices"].cpu().numpy().ravel(), minlength=16)
            n += len(patches)
    if not np.isfinite(total):
        raise ValueError("Non-finite validation loss")
    return {"pairs": n, "reconstruction_mse": total/n, "copy_initial_mse": copy_total/n,
            "code_counts": histogram.tolist(), "distinct_codes": int((histogram > 0).sum())}


def check_model(args, model, patches):
    # Independent expression of the original two-frame encoder/decode path.
    endpoints = patches[:, [0, -1]]
    with torch.no_grad():
        slots = model.action_latent.expand(len(patches), 2, -1, -1)
        z = model.to_codebook(model.encoder(torch.cat((slots, endpoints), dim=2))[:, 1:, :4])
        zq, _, _, ids = model.vq(z.reshape(len(patches), 4, 128))
        decoded = model.decoder(torch.cat((model.action_up(zq[:, None]), model.patch_up(endpoints[:, :1])), dim=2))[:, :, -256:]
        candidate = model.reconstruct_window(endpoints)
        torch.testing.assert_close(candidate["z_q"], zq[:, None], rtol=0, atol=0)
        torch.testing.assert_close(candidate["indices"], ids, rtol=0, atol=0)
        torch.testing.assert_close(candidate["recon"], decoded, rtol=0, atol=0)
    before = frozen_hash(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    for selected in (endpoints, patches):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model.pilot_loss(model.reconstruct_window(selected))
        loss.backward()
        for name in ("action_latent", "to_codebook.weight", "encoder.ffn.1.weight"):
            gradient = dict(model.named_parameters())[name].grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                raise ValueError(f"Missing/invalid encoder gradient: {name}")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("Frozen parameter received a gradient")
        optimizer.step()
    if frozen_hash(model) != before:
        raise ValueError("Frozen weights changed")
    print("CHECK PASSED: exact two-frame compatibility; 2/4-frame backward passes; frozen weights unchanged.")


def train(args, config, manifest):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = load_model(args).configure_pilot()
    training = Windows(args.cache_dir, manifest, "train")
    validation = Windows(args.cache_dir, manifest, "validation")
    run_provenance = provenance(args)
    (args.output_dir / "run.json").write_text(json.dumps({
        "variant": args.variant, "config": config, "provenance": run_provenance,
        "cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
        "train_pairs": len(training), "validation_pairs": len(validation),
    }, indent=2))
    before = frozen_hash(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = np.random.default_rng(config["seed"])
    order, cursor, history = generator.permutation(len(training)), 0, []
    initial = validate(model, validation, args.variant, config["micro_batch_size"])
    history.append({"update": 0, "validation": initial})
    print("Initial validation:", initial, flush=True)
    effective_batch = config["micro_batch_size"] * config["accumulation_steps"]
    for update in range(1, config["updates"] + 1):
        # Fixed shuffled sample stream, independent of 2/4-frame model size.
        selected = []
        while len(selected) < effective_batch:
            take = min(effective_batch-len(selected), len(order)-cursor)
            selected.extend(order[cursor:cursor+take].tolist())
            cursor += take
            if cursor == len(order):
                order, cursor = generator.permutation(len(training)), 0
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(3)
        for first in range(0, effective_batch, config["micro_batch_size"]):
            patches = training.batch(selected[first:first+config["micro_batch_size"]], args.variant)
            output = model.reconstruct_window(patches)
            losses = model.pilot_loss(output, config["commitment_beta"])
            if not all(torch.isfinite(v).item() for v in losses):
                raise ValueError("Non-finite loss")
            (losses[0] / config["accumulation_steps"]).backward()
            totals += np.array([v.detach().item() for v in losses]) / config["accumulation_steps"]
            del output, losses, patches
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        row = {"update": update, "training_total": totals[0], "training_reconstruction": totals[1],
               "training_commitment": totals[2], "gradient_norm_before_clip": float(norm)}
        if update % config["validation_every"] == 0 or update == config["updates"]:
            row["validation"] = validate(model, validation, args.variant, config["micro_batch_size"])
            adapted = {name: p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
            torch.save({"adapted_parameters": adapted, "update": update, "variant": args.variant,
                        "config": config, "base_checkpoint_sha256": RELEASE_HASH}, args.output_dir / "latest.pt")
            print(f"Update {update}: train MSE={totals[1]:.6f}; validation={row['validation']['reconstruction_mse']:.6f}", flush=True)
        history.append(row)
        with (args.output_dir / "history.jsonl").open("a") as log:
            log.write(json.dumps(row)+"\n")
    after = frozen_hash(model)
    if before != after:
        raise ValueError("Frozen weights changed")
    report = {"variant": args.variant, "config": config, "provenance": run_provenance,
        "cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
        "train_pairs": len(training), "validation_pairs": len(validation),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_weight_sha256": before, "frozen_weights_unchanged": True,
        "checkpoint_selection": "predeclared final update; no test access",
        "training": "single-view reconstruction + commitment; no numerical action supervision",
        "history": history}
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))
    print("Pilot completed:", args.output_dir / "results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cache", "check", "train"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dino-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("experiments/temporal_pilot/config.json"))
    parser.add_argument("--variant", choices=("two_frame", "four_frame"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-episodes", type=int, default=0, help="Per split; 0 means all. Smoke checks only.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Run inside an Euler GPU allocation")
    if args.batch_size < 1 or args.limit_episodes < 0:
        parser.error("Invalid batch size or episode limit")
    torch.set_num_threads(1)
    if args.command == "cache":
        if args.sample_dir is None or args.output_dir is None:
            parser.error("cache requires --sample-dir and --output-dir")
        cache_features(args)
        return
    if args.cache_dir is None:
        parser.error("check/train requires --cache-dir")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["provenance"]["checkpoint_sha256"] != RELEASE_HASH:
        raise ValueError("Invalid/incomplete cache")
    # Verify cached input files before fitting; no silent reuse of changed arrays.
    for entry in manifest["episodes"]:
        for file_key, hash_key in (("patch_file", "patch_sha256"), ("window_file", "window_sha256")):
            if sha256(args.cache_dir / entry[file_key]) != entry[hash_key]:
                raise ValueError(f"Cache integrity mismatch: {entry[file_key]}")
    config = read_config(args.config)
    if args.command == "check":
        model = load_model(args).configure_pilot()
        data = Windows(args.cache_dir, manifest, "train")
        check_model(args, model, data.batch([0], "four_frame"))
    else:
        if args.variant is None or args.output_dir is None:
            parser.error("train requires --variant and --output-dir")
        train(args, config, manifest)


if __name__ == "__main__":
    main()
