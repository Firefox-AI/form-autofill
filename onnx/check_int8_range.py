#!/usr/bin/env python
"""Fail if an int8-quantized ONNX model has weights outside the 7-bit range.

Per-channel weight quantization maps every output channel's maximum to full-scale
+-127. On x86-64 CPUs with AVX2 but without VNNI, ONNX Runtime's MLAS kernel
accumulates adjacent uint8 x int8 products into an int16 before widening to
int32 (VPMADDUBSW). Two full-scale products reach 2*255*127 = 64770, which
silently saturates at 32767 and corrupts the result. Quantizing with
reduce_range=True caps weights at +-64, so the worst adjacent pair is
2*255*64 = 32640 and stays exact.

This checks the produced weights rather than trusting that --reduce_range was
passed, so a quantizer upgrade or a stray QUANT_EXTRA override cannot silently
reintroduce Bug 2064781.

    python check_int8_range.py model_quantized.onnx [more.onnx ...]

Exits non-zero if any INT8 initializer contains |w| > 64.
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import numpy_helper

# reduce_range caps signed weights at 7 bits.
LIMIT = 64


def check(path, verbose=False):
  """Return the list of (name, n_over, size, max_abs) for offending tensors."""
  model = onnx.load(path)
  offenders = []
  n_int8 = 0
  for init in model.graph.initializer:
    if init.data_type != onnx.TensorProto.INT8:
      continue
    n_int8 += 1
    w = numpy_helper.to_array(init).astype(np.int32)
    over = int(np.count_nonzero(np.abs(w) > LIMIT))
    if over:
      offenders.append((init.name, over, w.size, int(np.abs(w).max())))
    elif verbose:
      print(f"    ok   {init.name}: max |w| = {int(np.abs(w).max())}")

  if not n_int8:
    print(f"{path}: WARNING no INT8 initializers found -- not an int8 model?")
  return offenders, n_int8


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("models", nargs="+", help="Quantized .onnx file(s) to check.")
  ap.add_argument("-v", "--verbose", action="store_true",
                  help="Also print per-tensor maxima for passing tensors.")
  args = ap.parse_args()

  failed = False
  for path in args.models:
    offenders, n_int8 = check(path, args.verbose)
    if offenders:
      failed = True
      total_over = sum(o[1] for o in offenders)
      total_w = sum(o[2] for o in offenders)
      print(f"{path}: FAIL -- {len(offenders)}/{n_int8} INT8 tensors exceed "
            f"+-{LIMIT} ({total_over}/{total_w} weights, "
            f"{100.0 * total_over / total_w:.2f}%)")
      for name, over, size, mx in offenders:
        print(f"    {name}: {over}/{size} over limit, max |w| = {mx}")
      print("    Re-quantize with --reduce_range (see Bug 2064781).")
    else:
      print(f"{path}: OK -- {n_int8} INT8 tensors all within +-{LIMIT}")

  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
