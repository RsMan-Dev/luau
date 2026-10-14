# Luau GPU kernels (experimental fork)

Functions marked `@gpu` can run on the GPU with a single primitive:

```luau
local ok, err = gpu.run(fn, buffers, runs_per_buffer)
```

which, when it returns true, has the effect of

```luau
for _, b in buffers do
    for index = 0, runs_per_buffer - 1 do -- in an unspecified order, concurrently
        fn(b, index)
    end
end
```

`fn(b, index)` gets one buffer, which it reads and mutates, and an index, an exact integer from 0. Its return value is
ignored. The invocations run in parallel as a Vulkan compute shader lowered from the function's IR by native code
generation. Nothing ever runs in the VM instead: gpu.run either leaves every buffer with its results and returns true,
or leaves every buffer byte for byte as it was and returns false and why.

```luau
-- a time step, then positions and velocities
@gpu
local function move(b: buffer, i: number)
    local dt = buffer.readf32(b, 0)
    local o = 16 + i * 8
    buffer.writef32(b, o, buffer.readf32(b, o) + buffer.readf32(b, o + 4) * dt)
end

local ok, err = gpu.run(move, { particles }, 100000)
```

The design goal is to stay as close to base Luau as possible: no new types, kernels are ordinary Luau functions over
ordinary `buffer`s that can also be called from Luau, and data layouts are written in Luau on top of the primitive.

## API

`openGpuLibrary` (`Gpu/include/Luau/GpuLibrary.h`) opens the library in a host's state:

| Function | Description |
|---|---|
| `gpu.run(fn, buffers, runs_per_buffer)` | runs `fn(b, index)` for `index = 0 .. runs_per_buffer-1` on each buffer `b` of the array; returns `true`, or `false, reason` |
| `gpu.atomicadd(b, offset, v)` | adds `v` to the u32 at `offset`, returns the previous value |
| `gpu.atomicmin(b, offset, v)` | unsigned 32-bit minimum, returns the previous value |

- `false, "Function cannot run in GPU"`: `fn` can't be lowered for the GPU: it isn't `@gpu`, uses what kernels don't
  support (see [Supported subset](#supported-subset)), or its environment isn't safe.
- `false, reason` for everything else that only shows when the run is attempted: no device or a refused one, a buffer
  over the device's limits, not enough memory, an error in the kernel, a run that was stopped or timed out. An error in
  the kernel gives the kind and source line of the run's first one, as in `the kernel failed on the GPU: integer too
  large for a float at line 980`: out of bounds or unaligned buffer access, integer overflow, integer division by zero,
  integer too large for a float, number out of the range of 32-bit integers, float overflow, number beyond the range of
  floats, loop limit reached, or a value of a type the kernel doesn't handle there.
- `false, "the kernel is being compiled for the GPU"`: the driver is still creating the kernel's pipeline, on a host
  thread, and the buffers are left as they were. Drivers take a time growing faster than the size of the kernel: tens of
  milliseconds for small ones, but about 40 s for 600 KB of GLSL and 160 s for 1.1 MB on NVIDIA's (Find a Geode's frame
  kernel), which a script can't wait for. Several pipelines are created at once. A script that needs the result calls
  `gpu.run` again (the examples do); `VulkanOptions::waitForPipelines` and `luau-gpu --wait-for-pipelines` make runs
  wait instead. A kernel of more than 2 MB of GLSL is refused, with "Function cannot run in GPU", since its pipeline
  would take a compile thread for most of an hour. Each kernel has two variants: the one that runs first only notes
  that a check failed, that a division or square root needs the exact computation (denormal numbers, a hardware result
  too far off), or that a float result is infinite, and a run where it did runs again on the checked variant, compiled
  the first time it's needed (a run can then return "being compiled" again), which gives the results and failures this
  contract describes. The checks and the exact computations otherwise make kernels twice as slow, in the registers they
  keep even when they don't run. The 64 runs after one that needed it start on the checked variant.
- Arguments of the wrong type or out of range (`runs_per_buffer` must be an integer from 0 to 2^31-1, `buffers` an array
  of buffers) are mistakes in the script and raise errors, as in other libraries.

`luau-gpu` keeps the pipelines the driver compiled in `~/.cache/luau-gpu/pipelines.bin` (`$XDG_CACHE_HOME/luau-gpu`
when set, `--no-pipeline-cache` to leave it out), so that a script runs its kernels at once from its second run: Find a
Geode's frame kernel takes 165 s on its first run without any cache, 5 s for the whole run with it. The library leaves it
to the host: `VulkanOptions::pipelineCachePath`, empty by default.

A buffer listed several times runs several times, in the order of the list; different buffers run concurrently.

## Contract

- Each invocation must only write words of its buffer that no other invocation of the run reads or writes, unless it
  uses the atomics: invocations run concurrently, and plain writes that race give an unspecified result on the GPU.
  Atomics are what kernels use to combine results from several invocations: counters, histograms, depth tests. Called
  from Luau, they are plain reads and writes.
- Numbers are 32-bit on the GPU. Integers stay exact: the lowering types each value as `int`, `uint` or `float` from how
  it's produced, and a result that wouldn't be the VM's (an integer overflow, an integer a float can't hold) fails the
  run. Whether a float holds an integer is a matter of its significant bits, not its magnitude: `2^26` and `0xffffff00`
  convert, `16777217` doesn't. Numbers convert only where an operation needs another class: a register holding an
  integer on a path and a float on another keeps each as it is, which joining the paths doesn't convert, and comparisons
  between integers and floats, or ints and uints, are exact. Where the VM rounds the integer once, the GPU does too, so
  it needn't be exact there: stored as a float (`buffer.writef32`, a vector component), or multiplied or divided by a
  power of two (`k / 2^27`). Vectors give the VM's bits: the VM computes them in f32 too, and the GPU rounds every
  operation the same way, divisions and square roots included (computed with integers, since Vulkan only bounds their
  error), without reordering them, and keeping NaN, infinities, signed zeros and denormal numbers. Other floating point
  math is f32 instead of the VM's f64, with every operation correctly rounded (division and square roots included):
  results differ from the VM's by f32 rounding. Float literals round to f32 like any other result (`1e30`, `0.1`). A
  result whose class f32's narrower range changes fails the run when it overflows: an infinity from finite operands
  (`1e30 * 1e10`, `math.exp(100)`, a literal beyond f32's range), which f64 would have kept finite. Underflows don't: a
  product or quotient that f32 rounds to 0 or a denormal number where f64 has a tiny normal number stays a result, since
  legitimate kernels produce them all the time (a falloff, `x ^ 40` of a small `x`; failing them failed the rasterizer's
  shading). A later step can then still differ from the VM's in kind, as dividing by that 0, which is the one way scalar
  float results can be categorically different from the VM's.
- Buffers stay on the GPU between runs. A run only uploads the 64 KB parts of a buffer that changed since the GPU last had
  it, and only downloads the parts its kernel wrote: kernels mark each part they write to, and the parts they don't are
  still those of the Luau buffer.

## Safety

A kernel can only touch the buffer it's given, and a run either completes cleanly on the GPU or changes nothing:

- every buffer access and atomic is checked against the real buffer size, as native code does; a bad access, an
  integer division by zero, `INT_MIN // -1` or another operation without the VM's result sets the status word, and the
  run fails
- a run going on for more than `VulkanOptions::abortMs` (100 ms by default, counted from when it starts on the GPU) is
  asked to stop: invocations that start afterwards give up at once and loops within 256 iterations, so a runaway kernel
  gives the GPU back quickly. Loops also count against a per-invocation budget (`VulkanOptions::loopBudget`, 2^24 by
  default)
- generated identifiers never come from user names, so nothing in a script can inject code or clash with reserved words
  of GLSL, GLSL ES, MSL or HLSL
- lowering, GLSL to SPIR-V compilation, SPIR-V validation, pipeline creation, device limits (storage buffer range,
  workgroup count, push constant size), allocation, submission and a zero status word must all succeed
- devices are checked when the backend is created: a probe kernel must keep NaN, infinities, signed zeros and denormal
  numbers, and see the host asking it to stop. Feature bits aren't trusted either way: llvmpipe reports
  SignedZeroInfNanPreserve and folds `inf - inf` to 0, and NVIDIA reports no DenormPreserve for f32 yet keeps denormal
  numbers when shaders declare it (and flushes them when they don't). Shaders always declare both modes, and a device
  whose probe doesn't behave as they require is refused
- a run that doesn't finish within `VulkanOptions::timeoutMs` (5 s by default) or a lost device disables the backend for
  good: its results are never copied back and every later run fails
- the backend is thread-safe

## Other platforms

Kernels are compiled once to SPIR-V, which is then cross-compiled with SPIRV-Cross. `ShaderCompiler`
(`Gpu/include/Luau/GpuShaderCompiler.h`) produces:

| Target | Output | Validated here |
|---|---|---|
| Vulkan 1.0 (desktop, Android) | SPIR-V 1.0 | SPIRV-Tools validator |
| OpenGL ES 3.1 (Android) | GLSL ES 3.10 compute shader | recompiled with glslang |
| Metal (iOS 13+, macOS 10.15+) | MSL 2.2, entry point `main0` | no: Apple's Metal compiler only exists on Apple platforms |

Resources follow `Gpu/include/Luau/GpuAbi.h` on every target; on Metal, the buffer is `[[buffer(0)]]`, the status word
`[[buffer(1)]]` and the push constants `[[buffer(2)]]`. Only the Vulkan backend exists; a Metal or OpenGL ES backend runs
these shaders the same way. `VulkanOptions::minimumLimits` simulates a device that only has the limits Vulkan guarantees,
as on low-end Android hardware.

## Supported subset

Inside `@gpu` functions, and the functions they call, which the compiler inlines:

- locals, assignments, `if`, `while`, `repeat`, numeric `for`, `break`, `continue`, `return`, if-expressions
- numbers, booleans and `vector`s; arithmetic, comparisons, `and`/`or`/`not`
- `buffer.readf32/readu32/readi32`, `buffer.writef32/writeu32/writei32` on 4-byte aligned offsets, `buffer.len`
- the `math`, `bit32` and `vector` builtins native code generation inlines
- `gpu.atomicadd`, `gpu.atomicmin`
- top-level `local` numeric constants

Not supported: tables, strings, upvalues other than constants, calls that aren't inlined, 8/16-bit and f64 buffer
accesses. `luau-gpu --emit=glsl` reports why each function can't run.

## Embedding

The core only has what native code generation needs: the `@gpu` attribute (behind the `LuauGpuKernels` flag), the
`LPF_GPU_FUNCTION` proto flag, the compiler inlining calls in `@gpu` functions, and the lowering
(`CodeGen/include/Luau/CodeGenGpu.h`). The library and the backend are in the host:

```cpp
std::string error;
auto backend = Luau::Gpu::VulkanBackend::create(error); // nullptr if there is no usable device
luaL_openlibs(L);
Luau::Gpu::openGpuLibrary(L, backend.get(), error);       // without a backend, gpu.run returns false and 'error'
luaL_sandbox(L);                                         // kernels need the real builtins
```

## Layout

- `Ast`, `Compiler`, `Common/include/Luau/Bytecode.h`: the `@gpu` attribute, inlining, `LPF_GPU_FUNCTION`
- `CodeGen/src/CodeGenGpu.cpp`: lowers the IR of `@gpu` functions to GLSL compute shaders
- `Gpu/src/GpuLibrary.cpp`: the `gpu` library
- `Gpu/src/GpuVulkan.cpp`: Vulkan backend (`Luau.Gpu.Vulkan`); keeps a device copy of each buffer, which it frees with the
  Luau buffer, and uploads only the parts that changed
- `Gpu/src/GpuShaderCompiler.cpp`: SPIR-V compilation, validation and cross-compilation (`Luau.Gpu.Shaders`)
- `Gpu/cli/Main.cpp`: `luau-gpu`, a host that runs a script with the backend and a small window library (SDL2)
- `Gpu/examples/selftest.luau`: runs kernels on the GPU and from Luau and compares the results
- `Gpu/examples/rasterizer.luau`: a software rasterizer drawing a 1M triangle model in real time
- `Gpu/examples/soak.luau`: fresh buffers and kernels every frame, to check that memory and cost stay flat
- `Gpu/bench/`: native Luau against GPU benchmarks, see [Gpu/bench/README.md](bench/README.md)
- `tests/CodeGenGpu.test.cpp`: lowering tests (part of `Luau.Conformance`)
- `tests/GpuBackend.test.cpp`: gpu.run, shader compilation for every target, limits, failures, threads (only with
  `LUAU_BUILD_GPU`)

## Building

The Vulkan backend, the shader compilers and `luau-gpu` are behind `LUAU_BUILD_GPU` (off by default) and need Vulkan,
shaderc, SPIRV-Tools, SPIRV-Cross, glslang and SDL2. On NixOS, `Gpu/shell.nix` provides them:

```sh
nix-shell Gpu/shell.nix --run 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLUAU_BUILD_GPU=ON && ninja -C build luau-gpu'
./build/luau-gpu Gpu/examples/selftest.luau
./build/luau-gpu Gpu/examples/rasterizer.luau     # drag to rotate, wheel to zoom, G to switch between the GPU and Luau
./build/luau-gpu --vm script.luau                  # without the GPU: gpu.run returns false
```

`luau-gpu --emit=<glsl|spirv|msl-ios|msl-macos|gles> script.luau` prints the shaders of every kernel for a target
without running the script (see [Other platforms](#other-platforms)).

## Known limitations

- Every run is synchronous, and downloads every part its kernel wrote, even when a later run overwrites it.
- Tools built on the upstream parser (luau-lsp, StyLua, selene) report `@gpu` as an invalid attribute.
