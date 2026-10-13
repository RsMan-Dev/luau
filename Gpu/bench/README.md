# GPU benchmarks

`bench.luau` runs the same kernels three ways and checks that all of them produce the same results:

| Mode | What runs |
|---|---|
| native loop | the kernel called from a plain Luau `for` loop, compiled to native code: what a script does without `gpu.dispatch` |
| vm | `gpu.dispatch` without a GPU backend, which calls the kernel from C for every invocation |
| gpu | `gpu.dispatch` on the GPU, **including every transfer** (uploads of changed buffers, downloads of written buffers) |

```sh
./build/luau-gpu Gpu/bench/bench.luau            # every scenario
./build/luau-gpu Gpu/bench/bench.luau saxpy      # scenarios whose name contains "saxpy"
```

Each mode runs on fresh copies of the same inputs; the table reports the best time (up to 3 native runs, 10 GPU runs,
2 seconds per mode), plus the GPU median. The script fails if a result differs or if a dispatch that should run on the
GPU doesn't.

## Scenarios

| Scenario | Measures | Result check |
|---|---|---|
| saxpy | memory bound: one multiply-add per 8 bytes moved, 1M floats | f32, relative 1e-6 |
| tiny saxpy | the same kernel on 64 elements: the fixed cost of a dispatch | f32, relative 1e-6 |
| mandelbrot | compute bound: up to 256 iterations per pixel, 1024x1024, no input | iteration counts, up to 1% may differ |
| n-body | O(n²) reads over O(n) data: 4096 bodies, each reading all the others | f32, relative 1e-3 |
| game of life | a grid with neighbor reads, 10 double buffered steps (10 dispatches) | exact |
| histogram | 4M values into 256 bins with `gpu.atomicadd`, heavy contention | exact |
| particles | vector math with a branch, 1M particles, 24 bytes each read and written | f32, relative 1e-5 |
| blur 5x5 | a post-processing pass: 25 reads per pixel, 1024x1024 | f32, relative 1e-5 |

Tolerances come from f32 on the GPU against f64 in the VM: Mandelbrot iteration counts near the set's boundary depend
on rounding, and n-body sums 4096 terms.

## Results

2026-10-13, NVIDIA GeForce RTX 4080, AMD Ryzen 9 9950X3D, Release build:

| Scenario | Size | Native loop ms | VM ms | GPU ms | GPU median ms | Speedup vs native | Match |
|---|---|---:|---:|---:|---:|---:|---|
| saxpy | 1000000 floats | 15.320 | 21.730 | 2.240 | 2.390 | 6.8x | yes |
| tiny saxpy | 64 floats | 0.001 | 0.001 | 0.318 | 0.334 | 0.003x | yes |
| mandelbrot | 1024x1024, 256 iterations | 254.478 | 256.236 | 1.471 | 1.551 | 173.0x | yes |
| n-body | 4096 bodies | 343.154 | 343.510 | 1.233 | 1.360 | 278.3x | yes |
| game of life | 1024x1024, 10 steps | 412.688 | 472.797 | 8.179 | 9.515 | 50.5x | yes |
| histogram | 4000000 values, 256 bins | 117.109 | 137.034 | 4.552 | 4.856 | 25.7x | yes |
| particles | 1000000 particles | 28.390 | 33.186 | 6.180 | 7.393 | 4.6x | yes |
| blur 5x5 | 1024x1024 | 115.955 | 126.279 | 1.125 | 2.396 | 103.1x | yes |

What it shows:

- **Compute bound kernels win by two orders of magnitude** (Mandelbrot, n-body, blur): the transfers are small next to
  the work.
- **Memory bound kernels win much less** (saxpy, particles): most of the GPU time is copying buffers to the device and
  back. Keeping data on the GPU between dispatches would remove most of it.
- **Small dispatches lose**: about 0.3 ms of fixed cost per dispatch (command buffer, submission, waiting for the GPU)
  against about a microsecond of native code for the 64-element saxpy. Where the break-even lies depends on the kernel;
  these scenarios don't measure it.
- Every dispatch is synchronous, so a 10-step simulation pays 10 round trips (game of life).
- The native loop calls the kernel through a function value, as a generic loop would. For a trivial kernel like saxpy, a
  loop with the body written inline is faster still (about 3 ms for 1M elements), which makes the GPU gain about 1.3x.

These numbers were measured with the script running in a sandboxed state (`luaL_sandbox` + `luaL_sandboxthread`, like
the `luau` command). Without it, the VM disables the fast paths for builtin calls (`buffer.*`, `math.*`), and the native
loop is 1.1 (Mandelbrot, no builtin calls in its loop) to 19 times (particles) slower, which inflates the GPU speedups.
