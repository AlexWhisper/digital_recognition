#!/usr/bin/env python3
"""
Export trained MNIST MLP model to ONNX.

Usage examples:
  python export_onnx.py \
    --model-dir ./saved_models \
    --output ./saved_models/model.onnx \
    --opset 13 \
    --verify \
    --simplify

This script expects files saved by train_and_save.py:
  - model_structure.json (contains layer_sizes, learning_rate, ...)
  - model.pth (PyTorch state_dict)
"""

import argparse
import json
import os
from typing import Dict, Any, Optional

import numpy as np
import torch

from model import NeuralNetwork


def build_model_from_structure(structure_path: str, weights_path: str, device: Optional[str] = None) -> NeuralNetwork:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(structure_path, "r", encoding="utf-8") as f:
        info: Dict[str, Any] = json.load(f)

    layer_sizes = info["layer_sizes"]
    learning_rate = info.get("learning_rate", 0.001)

    model = NeuralNetwork(layer_sizes, learning_rate).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset: int = 13,
    dynamic_batch: bool = True,
    input_name: str = "input",
    output_name: str = "logits",
) -> None:
    dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def try_simplify(onnx_path: str) -> None:
    try:
        import onnx  # noqa: F401
        from onnxsim import simplify  # type: ignore

        model_opt, check = simplify(onnx_path)
        if not check:
            print("[simplify] ONNX model could not be validated after simplification; keeping original.")
            return
        model_opt.save(onnx_path)
        print("[simplify] Simplified ONNX model saved.")
    except ImportError:
        print("[simplify] onnxsim not installed. Skipping simplification. You can install via: pip install onnxsim")


def verify_onnx(onnx_path: str, model: torch.nn.Module, sample_input: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed. Skipping verification. Install via: pip install onnxruntime")
        return

    # PyTorch output
    with torch.no_grad():
        torch_out = model(sample_input).cpu().numpy()

    # ONNXRuntime output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # keep CPU for determinism
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: sample_input.cpu().numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    max_rel_diff = float(np.max(np.abs(torch_out - ort_out) / (np.abs(torch_out) + 1e-12)))

    ok = np.allclose(torch_out, ort_out, atol=atol, rtol=rtol)
    status = "PASSED" if ok else "FAILED"
    print(f"[verify] ONNX vs PyTorch: {status}. max_abs_diff={max_abs_diff:.6g}, max_rel_diff={max_rel_diff:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained MNIST MLP model to ONNX")
    parser.add_argument("--model-dir", type=str, default="./saved_models", help="Directory containing model_structure.json and model.pth")
    parser.add_argument("--structure-json", type=str, default=None, help="Path to model_structure.json (overrides --model-dir if set)")
    parser.add_argument("--weights", type=str, default=None, help="Path to model.pth (overrides --model-dir if set)")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX file path (default: <model-dir>/model.onnx)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--static", action="store_true", help="Export with static batch size (disable dynamic axes)")
    parser.add_argument("--batch", type=int, default=1, help="Dummy batch size when exporting with static axes")
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier after export")
    parser.add_argument("--verify", action="store_true", help="Run ONNX Runtime and compare outputs with PyTorch")

    args = parser.parse_args()

    model_dir = args.model_dir
    structure_path = args.structure_json or os.path.join(model_dir, "model_structure.json")
    weights_path = args.weights or os.path.join(model_dir, "model.pth")
    output_path = args.output or os.path.join(model_dir, "model.onnx")

    if not os.path.isfile(structure_path):
        raise FileNotFoundError(f"model_structure.json not found at: {structure_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"model.pth not found at: {weights_path}")

    print(f"Loading model structure from: {structure_path}")
    print(f"Loading weights from: {weights_path}")

    model = build_model_from_structure(structure_path, weights_path)

    # Determine input feature size from structure
    with open(structure_path, "r", encoding="utf-8") as f:
        info: Dict[str, Any] = json.load(f)
    input_features = int(info["layer_sizes"][0])

    batch_size = max(1, int(args.batch))
    dummy = torch.randn((batch_size, input_features), dtype=torch.float32, device=next(model.parameters()).device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Exporting to ONNX: {output_path} (opset={args.opset}, dynamic_batch={not args.static})")

    export_to_onnx(
        model=model,
        dummy_input=dummy,
        output_path=output_path,
        opset=args.opset,
        dynamic_batch=not args.static,
        input_name="input",
        output_name="logits",
    )

    print("Export complete.")

    if args.simplify:
        try_simplify(output_path)

    if args.verify:
        # Rebuild a CPU dummy for stable ORT execution if needed
        dummy_cpu = dummy.detach().cpu()
        verify_onnx(output_path, model.cpu(), dummy_cpu)


if __name__ == "__main__":
    main()



