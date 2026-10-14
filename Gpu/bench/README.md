# GPU benchmarks

`bench.luau` runs the same kernels two ways and checks that both produce the same results:

| Mode | What runs |
|---|---|
| native loop | the kernel called from a plain Luau `for` loop, compiled to native code: what a script does without `gpu.run` |
| gpu | `gpu.run` on the GPU, **including every transfer** (uploads of the parts that changed, downloads of the parts written) |

```sh
./build/luau-gpu Gpu/bench/bench.luau            # every scenario
./build/luau-gpu Gpu/bench/bench.luau saxpy      # scenarios whose name contains "saxpy"
```

A kernel only gets its buffer, so each scenario keeps its inputs and its results in one buffer. Each mode runs on fresh
copies of the same buffer; the table reports the best time (up to 3 native runs, 10 GPU runs, 2 seconds per mode), plus
the GPU median. The script fails if a result differs or if a run fails.

## Scenarios

| Scenario | Measures | Result check |
|---|---|---|
| saxpy | 1000000 floats | 14.715 | 1.026 | 14.3x | yes |
| tiny saxpy | 64 floats | 0.001 | 0.022 | 0.05x | yes |
| mandelbrot | 1024x1024, 256 iterations | 254.000 | 1.075 | 236x | yes |
| n-body | 4096 bodies | 340.761 | 1.767 | 193x | yes |
| game of life | 1024x1024, 10 steps | 422.471 | 7.143 | 59.1x | yes |
| histogram | 4000000 values, 256 bins | 120.938 | 2.425 | 49.9x | yes |
| particles | 1000000 particles | 28.626 | 3.818 | 7.5x | yes |
| blur 5x5 | 1024x1024 | 117.874 | 1.064 | 111x | yes |

Tolerances come from f32 on the GPU against f64 in Luau: Mandelbrot iteration counts near the set's boundary depend on
rounding, and n-body sums 4096 terms.

## Results

2026-10-14, NVIDIA GeForce RTX 4080 (shared with another application using about 40% of it), AMD Ryzen 9 9950X3D,
RelWithDebInfo build, scripts sandboxed with native code generation (as `luau --codegen` runs them):

| Scenario | Size | Native loop ms | GPU ms | Speedup vs native | Match |
|---|---|---:|---:|---:|---|
| saxpy | 1000000 floats | 14.715 | 1.026 | 14.3x | yes |
| tiny saxpy | 64 floats | 0.001 | 0.022 | 0.05x | yes |
| mandelbrot | 1024x1024, 256 iterations | 254.000 | 1.075 | 236x | yes |
| n-body | 4096 bodies | 340.761 | 1.767 | 193x | yes |
| game of life | 1024x1024, 10 steps | 422.471 | 7.143 | 59.1x | yes |
| histogram | 4000000 values, 256 bins | 120.938 | 2.425 | 49.9x | yes |
| particles | 1000000 particles | 28.626 | 3.818 | 7.5x | yes |
| blur 5x5 | 1024x1024 | 117.874 | 1.064 | 111x | yes |

(medians of the best time of three runs of the script)

What it shows:

- **Compute bound kernels win by two orders of magnitude** (Mandelbrot, n-body, blur): the transfers are small next to
  the work. Exactness has a cost there: n-body divides vectors 3 times per pair of bodies, and divisions and square roots
  that give the VM's bits take it from 0.6 ms with the hardware's approximate ones to 1.8 ms; Mandelbrot's scalar
  products are checked for f32 overflows, from 0.8 to 1.1 ms.
- **Memory bound kernels win much less** (saxpy, particles): most of the GPU time is copying what the kernel wrote back.
  Only the 64 KB parts a kernel writes come back, so inputs in the same buffer stay on the GPU: saxpy downloads y and not
  x, the histogram its 1 KB of bins and not its 16 MB of values, the game of life the grid each step writes.
- **Uploads are cheap when little changes**: only the parts of a buffer that differ from the GPU's copy are sent, so the
  game of life, which only changes its header word between steps, uploads its grids once.
- **Small runs lose**: a few tens of µs of fixed cost per run (command buffer, submission, waiting for the GPU) against
  about a microsecond of native code for the 64-element saxpy. A run that writes more than 64 KB of a buffer costs a
  second submission, to download the parts once the host knows which ones were written.
- The native loop calls the kernel through a function value, as a generic loop would. For a trivial kernel like saxpy, a
  loop with the body written inline is faster still, which makes the GPU gain smaller.

These numbers were measured with the script running in a sandboxed state (`luaL_sandbox` + `luaL_sandboxthread`, like
the `luau` command). Without it, kernels can't run on the GPU, and the VM disables the fast paths for builtin calls
(`buffer.*`, `math.*`), which makes the native loop several times slower.
