"""
Export NNUE v2 weights to binary format for C++ engine.

Binary format (all float32, row-major):
  feature_map: int32 (2, 8, 90) — SQ_TO_FEAT table
  num_features: int32 — 692
  accumulator.weight: (128, 692)
  accumulator.bias: (128,)
  fc1.weight: (32, 256)
  fc1.bias: (32,)
  fc2.weight: (32, 32)
  fc2.bias: (32,)
  fc_out.weight: (32,)
  fc_out.bias: scalar

Usage:
    python export_weights_v2.py [checkpoint_path] [output_path]
"""

import sys
import os
import struct
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from nnue_net_v2 import NNUENetV2, SQ_TO_FEAT, FEATURES_PER_PERSPECTIVE


def export_weights(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = NNUENetV2()
    model.load_state_dict(ckpt['model_state_dict'])

    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    with open(output_path, 'wb') as f:
        # Feature map table: int32 (2, 8, 90)
        feat_map = SQ_TO_FEAT.astype(np.int32)
        f.write(feat_map.tobytes())
        print(f"  feature_map: {feat_map.shape} (int32)")

        # Number of features: int32
        f.write(struct.pack('i', FEATURES_PER_PERSPECTIVE))
        print(f"  num_features: {FEATURES_PER_PERSPECTIVE}")

        # Accumulator weight: (128, 692)
        w = model.accumulator.weight.detach().numpy()
        f.write(w.tobytes())
        print(f"  acc_weight: {w.shape}")

        # Accumulator bias: (128,)
        b = model.accumulator.bias.detach().numpy()
        f.write(b.tobytes())
        print(f"  acc_bias: {b.shape}")

        # FC1 weight: (32, 256)
        w = model.fc1.weight.detach().numpy()
        f.write(w.tobytes())
        print(f"  fc1_weight: {w.shape}")

        # FC1 bias: (32,)
        b = model.fc1.bias.detach().numpy()
        f.write(b.tobytes())
        print(f"  fc1_bias: {b.shape}")

        # FC2 weight: (32, 32)
        w = model.fc2.weight.detach().numpy()
        f.write(w.tobytes())
        print(f"  fc2_weight: {w.shape}")

        # FC2 bias: (32,)
        b = model.fc2.bias.detach().numpy()
        f.write(b.tobytes())
        print(f"  fc2_bias: {b.shape}")

        # FC out weight: (1, 32) -> (32,)
        w = model.fc_out.weight.detach().numpy().flatten()
        f.write(w.tobytes())
        print(f"  out_weight: {w.shape}")

        # FC out bias: scalar
        b = model.fc_out.bias.detach().numpy()
        f.write(b.tobytes())
        print(f"  out_bias: {b.shape}")

    file_size = os.path.getsize(output_path)
    print(f"\nExported to {output_path} ({file_size:,} bytes)")


if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        SCRIPT_DIR, 'checkpoints', 'nnue_v2_best.pt')
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        SCRIPT_DIR, 'checkpoints', 'nnue_v2_weights.bin')
    export_weights(ckpt_path, out_path)
