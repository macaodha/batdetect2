"""Export a BatDetect2 detector to ONNX, with the metadata needed to run it.

The network is exported on its own: audio loading, the spectrogram (STFT, PCEN,
spectral mean subtraction) and decoding of the output maps stay outside, as in
the Python package. The time axis is dynamic, so any clip length works, but the
caller must pad the spectrogram width to a multiple of ``divide_factor`` (see
``batdetect2_onnx.json``) and crop the outputs back. The export is checked
against PyTorch on real recordings at several clip lengths.

Usage:
    uv run --with onnx --with onnxruntime python .github/scripts/export_onnx.py \
        --output dist/onnx [--checkpoint PATH_OR_URI] [--opset 17]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from batdetect2.api_v2 import BatDetect2API


class DetectorOutputs(torch.nn.Module):
    """Wraps the detector so the ONNX graph has three plain tensor outputs."""

    def __init__(self, detector: torch.nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, spec: torch.Tensor):
        out = self.detector(spec)
        return out.detection_probs, out.size_preds, out.class_probs


def pad_time(spec: torch.Tensor, factor: int) -> torch.Tensor:
    width = spec.shape[-1]
    extra = -width % factor
    return torch.nn.functional.pad(spec, (0, extra)) if extra else spec


def as_batch(spec: torch.Tensor) -> torch.Tensor:
    return spec.reshape(1, 1, spec.shape[-2], spec.shape[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint path or URI; default: the bundled model",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--check-audio", type=Path, default=Path("example_data/audio")
    )
    ap.add_argument(
        "--check-durations",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0],
    )
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    api = BatDetect2API.from_checkpoint(args.checkpoint)
    detector = api.model.detector.eval()
    factor = int(detector.backbone.divide_factor)
    height = int(api.model_config.architecture.input_height)
    net = DetectorOutputs(detector).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "batdetect2.onnx"
    axes = {"batch": 0, "time": 3}
    torch.onnx.export(
        net,
        (torch.zeros(1, 1, height, 64 * factor),),
        str(model_path),
        input_names=["spec"],
        output_names=["detection", "size", "classes"],
        dynamic_axes={
            name: {i: a for a, i in axes.items()}
            for name in ["spec", "detection", "size", "classes"]
        },
        opset_version=args.opset,
        dynamo=False,
    )
    onnx.save(
        onnx.load(str(model_path)), str(model_path)
    )  # one file, no external data

    # Check against PyTorch on real audio, clip by clip, at several clip lengths.
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    files = sorted(args.check_audio.glob("*.wav"))
    if not files:
        print(
            f"no .wav files in {args.check_audio} to check against",
            file=sys.stderr,
        )
        return 1
    worst = {}
    for duration in args.check_durations:
        diff = 0.0
        for f in files:
            wav = api.load_audio(f)
            n = int(round(duration * api.audio_config.samplerate))
            for start in range(0, len(wav), n):
                clip = wav[start : start + n]
                clip = np.pad(clip, (0, n - len(clip)))
                spec = pad_time(
                    as_batch(api.generate_spectrogram(clip)), factor
                )
                with torch.no_grad():
                    ref = [t.numpy() for t in net(spec)]
                got = session.run(None, {"spec": spec.numpy()})
                diff = max(
                    diff,
                    max(
                        float(np.abs(a - b).max())
                        for a, b in zip(ref, got, strict=True)
                    ),
                )
        worst[str(duration)] = diff
    print("max abs difference ONNX vs PyTorch by clip length (s):", worst)
    if max(worst.values()) > args.tolerance:
        print(f"difference above {args.tolerance}", file=sys.stderr)
        return 1

    names = list(api.targets.class_names)
    labels = []
    for name in names:
        tags = api.targets.decode_class(name)
        labels.append(tags[0].value if tags else name)
    (args.output / "labels.txt").write_text("\n".join(labels) + "\n")

    meta = {
        "batdetect2_version": version("batdetect2"),
        "checkpoint": args.checkpoint or "default",
        "onnx_opset": args.opset,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input": {
            "name": "spec",
            "shape": ["batch", 1, height, "time"],
            "dtype": "float32",
            "note": "spectrogram as produced by the preprocessor; pad time to a multiple of divide_factor",
        },
        "outputs": {
            "detection": ["batch", 1, height, "time"],
            "size": ["batch", 2, height, "time"],
            "classes": ["batch", len(names), height, "time"],
        },
        "divide_factor": factor,
        "class_names": names,
        "labels": labels,
        "default_clip_duration": api.inference_config.clipping.duration,
        "model_config": api.model_config.model_dump(mode="json"),
        "audio_config": api.audio_config.model_dump(mode="json"),
        "inference_config": api.inference_config.model_dump(mode="json"),
        "check": {
            "max_abs_difference": worst,
            "files": [f.name for f in files],
        },
    }
    (args.output / "batdetect2_onnx.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(
        f"wrote {model_path} ({model_path.stat().st_size / 1e6:.1f} MB), labels.txt, batdetect2_onnx.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
