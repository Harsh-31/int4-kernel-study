"""Run ONE benchmark case so Nsight Compute (ncu) can profile it.

    ncu --profile-from-start off ... python -m analysis.profile_case --impl marlin --layer mlp_up --M 1

What this does:
  1. Builds the same random INT4 weights the benchmark uses, calls prepare(),
     and checks correctness first (we never profile a failing implementation).
  2. Warms up, so compilation, autotuning and first-call costs are over.
  3. Turns the profiler ON, calls run() exactly once, turns it OFF.

Because ncu is started with --profile-from-start off, it records only the GPU
kernels launched inside step 3, i.e. exactly the kernels of one run() call. For
most implementations that is one kernel; triton_kernel_v2 with split-K also
launches the small zero-fill and FP32 -> FP16 cast kernels, so they appear too.

Called by carc/jobs/profile_ncu.sh once per case; see that script.
"""

import argparse

import torch

from common.config import group_sizes, load_config, quant_scheme
from common.correctness import check
from common.harness import compute_dtype, discover_impls, prepare_input, reference_weights
from common.quantize import random_activations, random_quantized_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one case for Nsight Compute.")
    parser.add_argument("--impl", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--M", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("needs a GPU; run it through carc/jobs/profile_ncu.sh")

    cfg = load_config()
    shape = cfg["layers"][args.layer]
    K, N = shape["K"], shape["N"]
    G = group_sizes("early")[0]
    scheme = quant_scheme()
    impls = discover_impls()
    if args.impl not in impls:
        raise SystemExit(f"[profile] implementation '{args.impl}' is not available")
    impl = impls[args.impl]

    with torch.inference_mode():
        qw = random_quantized_weights(K, N, G, scheme, device="cuda")
        impl.prepare(qw.q, qw.scales, qw.zeros, G)
        x = random_activations(args.M, K, device="cuda")
        x_in = prepare_input(impl, x)
        result = check(impl.run(x_in), x, reference_weights(args.impl, impl, qw),
                       compute_dtype(impl))
        if not result.passed:
            raise SystemExit(f"[profile] {args.impl} fails correctness; not profiling")

        for _ in range(cfg["runs"]["warmup"]):
            impl.run(x_in)
        torch.cuda.synchronize()

        torch.cuda.profiler.start()      # ncu starts recording here ...
        impl.run(x_in)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()       # ... and stops here

    print(f"[profile] {args.impl} {args.layer} K={K} N={N} M={args.M} G={G}: profiled one run()")


if __name__ == "__main__":
    main()
