"""Matched action probes for the released, two-frame and four-frame encoders.

Extraction uses GPU; fitting uses CPU. Only train/validation data are accessed.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.temporal_pilot import RELEASE_HASH, Windows, frozen_hash, load_model, provenance, sha256


VARIANTS = ("released", "two_frame", "four_frame")
SEEDS = (42, 43, 44, 45)
GROUPS = {"delta_x": [0], "delta_y": [1], "delta_z": [2],
          "orientation_1": [3], "orientation_2": [4], "orientation_3": [5],
          "translation": [0, 1, 2], "orientation": [3, 4, 5],
          "gripper": [6], "movement_6": list(range(6)), "all_7": list(range(7))}


def group_scores(prediction, truth, baseline):
    """Inputs are already normalized using the shared training-derived scales."""
    scores = {}
    for name, dims in GROUPS.items():
        error = float(np.mean(np.sum((prediction[:, dims] - truth[:, dims]) ** 2, axis=1)))
        base_error = float(np.mean(np.sum((baseline[dims] - truth[:, dims]) ** 2, axis=1)))
        variance = float(np.var(truth[:, dims], axis=0).sum())
        scores[name] = {"nmse": error/variance if variance > 1e-12 else None,
            "baseline_nmse": base_error/variance if variance > 1e-12 else None,
            "error_reduction_percent": 100*(1-error/base_error) if base_error > 1e-12 else None}
    return scores


def fit_pca(training, validation):
    """Training-only PCA; no whitening and no validation fitting."""
    if training.ndim != 2 or training.shape[1] != 512 or len(training) < 128:
        raise ValueError("Need at least 128 training examples with 512D features")
    mean = training.astype(np.float64).mean(axis=0)
    _, singular, components = np.linalg.svd(training.astype(np.float64)-mean, full_matrices=False)
    components = components[:128]
    rank = int(np.sum(singular > singular[0]*1e-6)) if singular[0] else 0
    projected = {"train": (training.astype(np.float64)-mean) @ components.T,
                 "validation": (validation.astype(np.float64)-mean) @ components.T}
    return projected, mean, components, singular, rank


def overlay_adaptation(model, saved, expected_variant):
    if saved["variant"] != expected_variant or saved["base_checkpoint_sha256"] != RELEASE_HASH:
        raise ValueError("Wrong adapted checkpoint variant/base")
    if saved["update"] != saved["config"]["updates"]:
        raise ValueError("Checkpoint is not the predeclared final update")
    expected = {name: p for name,p in model.named_parameters() if p.requires_grad}
    if set(saved["adapted_parameters"]) != set(expected):
        raise ValueError("Adapted parameter set does not match trainable encoding parameters")
    with torch.no_grad():
        for name, value in saved["adapted_parameters"].items():
            if value.shape != expected[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adapted parameter: {name}")
            expected[name].copy_(value)


def extract(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Extraction requires a GPU allocation")
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["offsets"] != [0, 3, 6, 9]:
        raise ValueError("Need a completed fixed-horizon temporal cache")
    if manifest["limit_episodes_per_split"]:
        raise ValueError("Use the full cache, not the smoke sample")
    if sha256(args.sample_dir / "manifest.json") != manifest["sample_manifest_sha256"]:
        raise ValueError("V3 sample does not match cached preparation")
    normalization = args.sample_dir / "action_normalization.npz"
    with np.load(normalization, allow_pickle=False) as norm:
        if int(norm["horizon"]) != 9 or not np.all(norm["net_scale"] > 0):
            raise ValueError("Invalid horizon-aware normalization")
    for entry in manifest["episodes"]:
        if entry["split"] not in ("train", "validation"):
            raise ValueError("Cache must contain only train/validation episodes")
        for file_key, hash_key in (("patch_file", "patch_sha256"), ("window_file", "window_sha256")):
            if sha256(args.cache_dir / entry[file_key]) != entry[hash_key]:
                raise ValueError(f"Changed cache file: {entry[file_key]}")
    source_sets = {s: {e["source_path"] for e in manifest["episodes"] if e["split"] == s}
                   for s in ("train", "validation")}
    if source_sets["train"] & source_sets["validation"]:
        raise ValueError("Episode source leakage")
    directories = {"two_frame": args.two_frame_dir, "four_frame": args.four_frame_dir}
    training_reports = {v: json.loads((d / "results.json").read_text()) for v,d in directories.items()}
    if training_reports["two_frame"]["config"] != training_reports["four_frame"]["config"]:
        raise ValueError("Adaptation training configurations differ")
    cache_hash = sha256(args.cache_dir / "manifest.json")
    for variant, report in training_reports.items():
        if report["variant"] != variant or not report["frozen_weights_unchanged"]:
            raise ValueError("Incomplete or wrong training report")
        if report["cache_manifest_sha256"] != cache_hash:
            raise ValueError("Training used a different temporal cache")
    args.output_dir.mkdir(parents=True)
    shutil.copy2(normalization, args.output_dir / normalization.name)
    # Build identical target and identity arrays in the same order as Windows.
    identities = {}
    for split in ("train", "validation"):
        fields = {name: [] for name in ("targets_raw", "record_indices", "frame_indices")}
        for entry in manifest["episodes"]:
            if entry["split"] != split:
                continue
            with np.load(args.cache_dir / entry["window_file"], allow_pickle=False) as saved:
                fields["targets_raw"].append(saved["targets_raw"])
                fields["frame_indices"].append(saved["original_frame_indices"])
                fields["record_indices"].append(np.full(len(saved["windows"]), entry["record_index"], dtype=np.int64))
        identities[split] = {key: np.concatenate(values) for key,values in fields.items()}
    report = {"scope": __doc__, "provenance": provenance(args), "cache_manifest_sha256": cache_hash,
              "normalization_sha256": sha256(normalization), "variants": {}, "sample_manifest_sha256": manifest["sample_manifest_sha256"]}
    for variant in VARIANTS:
        model = load_model(args).configure_pilot()
        if variant != "released":
            checkpoint_path = directories[variant] / "latest.pt"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved["config"] != training_reports[variant]["config"]:
                raise ValueError("Checkpoint/report configuration mismatch")
            overlay_adaptation(model, saved, variant)
            if frozen_hash(model) != training_reports[variant]["frozen_weight_sha256"]:
                raise ValueError("Frozen base weights differ from the adaptation run")
            del saved
            report["variants"][variant] = {"adapted_checkpoint_sha256": sha256(checkpoint_path),
                "training_report": training_reports[variant]}
        else:
            report["variants"][variant] = {"base_checkpoint_sha256": RELEASE_HASH}
        model.requires_grad_(False)
        destination = args.output_dir / variant
        destination.mkdir()
        for split in ("train", "validation"):
            dataset = Windows(args.cache_dir, manifest, split)
            features, indices = [], []
            input_variant = "four_frame" if variant == "four_frame" else "two_frame"
            with torch.no_grad():
                for first in range(0, len(dataset), args.batch_size):
                    last = min(first+args.batch_size, len(dataset))
                    encoded = model.encode_window(dataset.batch(range(first,last), input_variant))
                    values = encoded["z_q"].flatten(1)
                    if values.shape != (last-first, 512) or not torch.isfinite(values).all():
                        raise ValueError("Invalid extracted representation")
                    features.append(values.cpu().numpy())
                    indices.append(encoded["indices"].cpu().numpy())
                    if last % (args.batch_size*250) == 0:
                        print(f"{variant}/{split}: {last}/{len(dataset)}", flush=True)
            if len(dataset) != len(identities[split]["targets_raw"]):
                raise ValueError("Feature/label ordering mismatch")
            np.savez(destination / f"{split}.npz", features=np.concatenate(features),
                     indices=np.concatenate(indices), **identities[split])
            print(f"Saved {variant}/{split}: {len(dataset)} features", flush=True)
            del dataset
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    print("Extraction complete. No test data or fitting.")


def fit(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    manifest = json.loads((args.features_dir / "manifest.json").read_text())
    normalization = args.features_dir / "action_normalization.npz"
    if sha256(normalization) != manifest["normalization_sha256"]:
        raise ValueError("Normalization changed")
    with np.load(normalization, allow_pickle=False) as norm:
        mean, scale = norm["net_mean"], norm["net_scale"]
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid normalization")
    args.output_dir.mkdir(parents=True)
    report = {"scope": "matched train/validation probes; no test evaluation",
        "feature_manifest": manifest, "probe_seeds": SEEDS,
        "protocol": "per-model training-only PCA128; Linear128->7; Adam .001, 30 epochs, batch512, cosine to0; no clipping/weight decay; final epoch",
        "reference": "training-only least squares, bias, rcond1e-6; separate diagnostic",
        "uncertainty": "probe optimization seeds only; one LAM adaptation seed and data split",
        "variants": {}}
    reference_identities = {}
    for variant in VARIANTS:
        data = {}
        for split in ("train", "validation"):
            with np.load(args.features_dir / variant / f"{split}.npz", allow_pickle=False) as saved:
                data[split] = {key: saved[key] for key in saved.files}
            if not all(np.isfinite(data[split][key]).all() for key in ("features", "targets_raw")):
                raise ValueError("Non-finite features/targets")
            for key in ("targets_raw", "record_indices", "frame_indices"):
                if variant == "released":
                    reference_identities[split, key] = data[split][key]
                else:
                    np.testing.assert_array_equal(data[split][key], reference_identities[split, key])
        if set(data["train"]["record_indices"]) & set(data["validation"]["record_indices"]):
            raise ValueError("Episode record leakage")
        x_np, center, components, singular, rank = fit_pca(data["train"]["features"], data["validation"]["features"])
        y_np = {s: (data[s]["targets_raw"].astype(np.float64)-mean)/scale for s in data}
        baseline = y_np["train"].mean(axis=0)
        x = {s: torch.tensor(values, dtype=torch.float32) for s,values in x_np.items()}
        y_train = torch.tensor(y_np["train"], dtype=torch.float32)
        destination = args.output_dir / variant
        destination.mkdir()
        predictions = {f"{s}_targets_raw": data[s]["targets_raw"] for s in data}
        per_seed = {s: [] for s in data}
        print(f"\n{variant}: PCA numerical rank {rank}", flush=True)
        for seed in SEEDS:
            torch.manual_seed(seed)
            probe = torch.nn.Linear(128, 7)
            optimizer = torch.optim.Adam(probe.parameters(), lr=.001, weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
            generator = torch.Generator().manual_seed(seed+100)
            for _ in range(30):
                for batch in torch.randperm(len(y_train), generator=generator).split(512):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.mse_loss(probe(x["train"][batch]), y_train[batch])
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite probe loss")
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            torch.save(probe.state_dict(), destination / f"seed_{seed}.pt")
            with torch.no_grad():
                for split in data:
                    prediction = probe(x[split]).numpy().astype(np.float64)
                    per_seed[split].append(group_scores(prediction, y_np[split], baseline))
                    predictions[f"{split}_seed_{seed}_raw"] = prediction*scale+mean
            print(f"  seed {seed}: validation NMSE={per_seed['validation'][-1]['all_7']['nmse']:.6f}", flush=True)
        design = np.column_stack((x_np["train"], np.ones(len(y_train))))
        weights, _, ols_rank, _ = np.linalg.lstsq(design, y_np["train"], rcond=1e-6)
        variant_report = {"pca_rank": rank, "least_squares_rank": int(ols_rank), "splits": {}}
        for split in data:
            ols = np.column_stack((x_np[split], np.ones(len(y_np[split])))) @ weights
            predictions[f"{split}_least_squares_raw"] = ols*scale+mean
            split_report = {"pairs": len(y_np[split]), "per_seed": per_seed[split],
                "least_squares": group_scores(ols, y_np[split], baseline), "summary": {}}
            for name in GROUPS:
                values = [row[name] for row in per_seed[split]]
                result = {"baseline_nmse": values[0]["baseline_nmse"]}
                for key in ("nmse", "error_reduction_percent"):
                    numbers = [v[key] for v in values]
                    result[key+"_mean"] = float(np.mean(numbers)) if None not in numbers else None
                    result[key+"_std"] = float(np.std(numbers, ddof=1)) if None not in numbers else None
                split_report["summary"][name] = result
            truth = data[split]["targets_raw"][:, 6]
            if np.all(np.isclose(truth,0) | np.isclose(truth,1)):
                accuracy = [float(np.mean((predictions[f"{split}_seed_{seed}_raw"][:,6] >= .5) == (truth >= .5))) for seed in SEEDS]
                split_report["gripper_accuracy"] = {"mean": float(np.mean(accuracy)), "sample_std": float(np.std(accuracy,ddof=1)),
                    "baseline": float(np.mean((data["train"]["targets_raw"][:,6].mean() >= .5) == (truth >= .5)))}
            variant_report["splits"][split] = split_report
        np.savez(destination / "preprocessing.npz", feature_mean=center, pca_components=components,
            singular_values=singular, target_mean=mean, target_scale=scale, least_squares_weights=weights)
        np.savez(destination / "predictions.npz", **predictions)
        report["variants"][variant] = variant_report
    print("\nVALIDATION: matched 30-epoch probes (mean +/- probe-seed SD)")
    print(f"{'Model':14s} {'All 7':>18s} {'Translation':>14s} {'Orientation':>14s} {'Gripper':>12s} {'OLS all7':>12s}")
    for variant in VARIANTS:
        result = report["variants"][variant]["splits"]["validation"]
        summary = result["summary"]
        all_seven = summary["all_7"]
        def display(value):
            return f"{value:.6f}" if value is not None else "undefined"
        print(f"{variant:14s} {display(all_seven['nmse_mean'])} +/- {display(all_seven['nmse_std'])} "
              f"{display(summary['translation']['nmse_mean']):>14s} {display(summary['orientation']['nmse_mean']):>14s} "
              f"{display(summary['gripper']['nmse_mean']):>12s} {display(result['least_squares']['all_7']['nmse']):>12s}")
        print("  Error reduction vs training-mean baseline (%): " + ", ".join(
            f"{name}={display(summary[name]['error_reduction_percent_mean'])}" for name in ("all_7", "translation", "orientation", "gripper")))
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))
    print("Saved:", args.output_dir / "results.json", "\nNo test data loaded or evaluated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser("extract")
    for name in ("cache-dir", "sample-dir", "checkpoint", "dino-dir", "two-frame-dir", "four-frame-dir", "output-dir"):
        extraction.add_argument("--"+name, type=Path, required=True)
    extraction.add_argument("--batch-size", type=int, default=4)
    fitting = sub.add_parser("fit")
    fitting.add_argument("--features-dir", type=Path, required=True)
    fitting.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "extract":
        if args.batch_size < 1:
            parser.error("Positive batch size required")
        extract(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
