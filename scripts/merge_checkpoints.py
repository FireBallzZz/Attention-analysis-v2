"""Merge saved checkpoints in time order.

The oldest checkpoint is the start. Each later checkpoint is averaged with
the merged weights so far, then the next one is merged the same way.

Official checkpoints/scb_yolo.pt is YOLOv7. It is a different network from
the YOLO11 runs, so it is not part of the merge and is not overwritten.
"""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"

# Official downloaded YOLOv7 weights. Different architecture from YOLO11.
SKIP_NAMES = {"scb_yolo.pt"}


def _family(path: Path) -> str | None:
    name = path.name
    if name in SKIP_NAMES or name.endswith(".tmp"):
        return None
    if name.startswith("fusion_"):
        return "fusion"
    if name.startswith("l2cs_"):
        return "l2cs"
    if name.startswith("scb_yolo"):
        return "scb"
    return None


def _ordered(family: str) -> list[Path]:
    paths = [p for p in CHECKPOINTS.iterdir() if p.is_file() and _family(p) == family]
    return sorted(paths, key=lambda p: (p.stat().st_mtime, p.name))


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if not isinstance(state, dict):
            state = state.state_dict()
    return {k.replace("module.", ""): v.detach().cpu().clone() for k, v in state.items()}


def _same_weights(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[k], right[k]) for k in left)


def _average(previous: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Average floating weights. Keep the newer value for counts and mismatched keys."""
    merged: dict[str, torch.Tensor] = {}
    for key, newer in current.items():
        older = previous.get(key)
        if (
            older is not None
            and torch.is_tensor(older)
            and torch.is_tensor(newer)
            and older.shape == newer.shape
            and older.is_floating_point()
            and newer.is_floating_point()
        ):
            merged[key] = ((older.float() + newer.float()) / 2.0).to(dtype=newer.dtype)
        else:
            merged[key] = newer
    for key, older in previous.items():
        if key not in merged:
            merged[key] = older
    return merged


def merge_family(family: str) -> tuple[dict[str, torch.Tensor], list[Path]]:
    paths = _ordered(family)
    if not paths:
        raise SystemExit(f"No {family} checkpoints in {CHECKPOINTS}")
    running = _state_dict(paths[0])
    used = [paths[0]]
    print(f"{family}: start {paths[0].name}")
    for path in paths[1:]:
        current = _state_dict(path)
        if _same_weights(running, current) or (
            used and _same_weights(_state_dict(used[-1]), current)
        ):
            print(f"{family}: skip {path.name} (same weights as the previous checkpoint)")
            continue
        running = _average(running, current)
        used.append(path)
        print(f"{family}: merged {path.name}")
    return running, used


def _save_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, path)
    print(f"Wrote {path}")


def _save_yolo(path: Path, state: dict[str, torch.Tensor], newest: Path) -> None:
    ckpt = torch.load(newest, map_location="cpu", weights_only=False)
    model = ckpt["model"]
    model.load_state_dict(state)
    ckpt["ema"] = None
    ckpt["optimizer"] = None
    ckpt["scaler"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"Wrote {path}")


def main() -> None:
    fusion, fusion_used = merge_family("fusion")
    _save_state(CHECKPOINTS / "fusion_merged.pt", fusion)

    gaze, gaze_used = merge_family("l2cs")
    _save_state(CHECKPOINTS / "l2cs_merged.pt", gaze)

    scb, scb_used = merge_family("scb")
    yolo_sources = [p for p in scb_used if p.suffix == ".pt"]
    _save_yolo(CHECKPOINTS / "scb_yolo11n_merged.pt", scb, yolo_sources[-1])

    print("Skipped checkpoints/scb_yolo.pt (official YOLOv7, different network).")
    print(f"Fusion sources: {', '.join(p.name for p in fusion_used)}")
    print(f"L2CS sources: {', '.join(p.name for p in gaze_used)}")
    print(f"SCB sources: {', '.join(p.name for p in scb_used)}")


if __name__ == "__main__":
    main()
