# Luau GPU kernels (experimental fork)

Functions marked `@gpu` can be dispatched to the GPU with a single primitive:

```luau
gpu.dispatch(fn, n, ...buffers)
```

which has the semantics of

```luau
for i = 0, n - 1 do fn(i, ...buffers) end -- in an unspecified order
```

When a GPU backend is installed and `fn` fits the supported subset, the invocations run in parallel as a Vulkan
compute shader. Otherwise, they run in the VM with identical results (up to floating point precision, see below).
`gpu.status(fn)` returns `"gpu"`, or `"vm"` and the reason why the function stays in the VM.

```luau
@gpu
local function move(i: number, pos: buffer, vel: buffer, params: buffer)
    local o = i * 4
    local dt = buffer.readf32(params, 0)
    buffer.writef32(pos, o, buffer.readf32(pos, o) + buffer.readf32(vel, o) * dt)
end

gpu.dispatch(move, 100000, positions, velocities, params)
```

The design goal is to stay as close to base Luau as possible: no new types, kernels are ordinary Luau functions over
ordinary `buffer`s, and higher level constructs (data layouts, double buffering, 2D grids) are written in Luau on top
of the primitive.

## API

| Function | Description |
|---|---|
| `gpu.dispatch(fn, n, ...buffers)` | runs `fn(i, ...buffers)` for `i = 0 .. n-1`; only buffers can be passed |
| `gpu.status(fn)` | `"gpu"`, or `"vm", reason` |
| `gpu.atomicadd(b, offset, v)` | adds `v` to the u32 at `offset`, returns the previous value |
| `gpu.atomicmin(b, offset, v)` / `gpu.atomicmax(b, offset, v)` | unsigned 32-bit min/max, returns the previous value |

Atomics work everywhere (they are plain read-modify-write operations in the VM) and are what kernels use to combine
results from several invocations: counters, histograms, reductions, depth tests.

## Contract

- The result must not depend on the order of invocations. Two invocations writing the same location without atomics
  give an undefined result on the GPU (the VM stays deterministic).
- Numbers are 32-bit on the GPU: the translator types each value as `int`, `uint` or `float` from how it is produced
  (integer literals, the invocation index, buffer offsets, `math.floor` and `bit32` give integers; `/`, `^` and `readf32`
  give floats). Floating point math is f32 instead of f64, and integer overflow wraps instead of losing precision.
- An out of bounds or unaligned buffer access, or an integer division by zero, discards the GPU results and reruns the
  whole dispatch in the VM, which raises the error Luau would.

## Safety

A kernel can only touch the buffers it is given, and a dispatch either completes cleanly on the GPU or runs in the VM:

- every buffer access and atomic is bounds and alignment checked against the real buffer size; a bad access, an
  integer division by zero, `INT_MIN // -1` or another operation that is undefined on the GPU sets the status word, the
  GPU results are discarded and the VM reruns the dispatch, raising the error Luau would
- a dispatch that runs for more than `VulkanOptions::abortMs` (100 ms by default, counted from when it starts on the GPU)
  is asked to stop: invocations that start afterwards give up at once and loops within 256 iterations, so a runaway kernel
  gives the GPU back quickly; it is then discarded like an error, and in the VM the host's interrupt callback applies as for
  any script. Loops also count against a per-invocation budget (`VulkanOptions::loopBudget`, 2^24 by default)
- generated identifiers never come from user names, so nothing in a script can inject code or clash with reserved words
  of GLSL, GLSL ES, MSL or HLSL
- the GPU path is all or nothing: translation, GLSL to SPIR-V compilation, SPIR-V validation, pipeline creation, device
  limits (storage buffers per stage, storage buffer range, workgroup count, push constant size), allocation, submission
  and a zero status word must all succeed, otherwise the VM runs the dispatch and `gpu.status` says why
- a dispatch that doesn't finish within `VulkanOptions::timeoutMs` (5 s by default) or a lost device disables the backend
  for good: its results are never copied back and every later dispatch runs in the VM
- the backend is thread-safe

## Other platforms

Kernels are compiled once to SPIR-V, which is then cross-compiled with SPIRV-Cross. `ShaderCompiler`
(`Gpu/include/Luau/GpuShaderCompiler.h`) produces:

| Target | Output | Validated here |
|---|---|---|
| Vulkan 1.0 (desktop, Android) | SPIR-V 1.0 | SPIRV-Tools validator |
| OpenGL ES 3.1 (Android) | GLSL ES 3.10 compute shader | recompiled with glslang |
| Metal (iOS 13+, macOS 10.15+) | MSL 2.2, entry point `main0` | no: Apple's Metal compiler only exists on Apple platforms |

Resources follow `Gpu/include/Luau/GpuAbi.h` on every target; on Metal, buffer *i* is `[[buffer(i)]]`, the status word
follows the buffers and the push constants are `[[buffer(16)]]`. Only the Vulkan backend exists; a Metal or OpenGL ES
backend implements `lua_GpuBackend` with these shaders. `VulkanOptions::minimumLimits` simulates a device that only has
the limits Vulkan guarantees, as on low-end Android hardware: kernels that need more storage buffers or larger buffers
then run in the VM.

## Supported subset

Inside `@gpu` functions:

- locals, assignments, compound assignments, `if`/`elseif`/`else`, `while`, `repeat`, numeric `for`, `break`,
  `continue`, `return`, if-expressions
- numbers, booleans and `vector`s; arithmetic, comparisons, `and`/`or`/`not` on booleans
- `buffer.readf32/readu32/readi32`, `buffer.writef32/writeu32/writei32`, `buffer.len` on 4-byte aligned offsets
- `math` (floor, ceil, round, abs, sign, min, max, clamp, sqrt, sin, cos, tan, asin, acos, atan, atan2, exp, log, pow,
  fmod, lerp, huge, pi), `bit32` (band, bor, bxor, bnot, lshift, rshift, arshift, btest, extract),
  `vector` (create, dot, cross, magnitude, normalize, floor, ceil, abs, min, max, `.x/.y/.z`)
- calls to other `@gpu` local functions (specialized per argument types, one return value, no recursion)
- top-level `local` numeric constants

Not supported (the function then runs in the VM): tables, strings, closures and upvalues other than numeric constants,
multiple return values, generic `for`, 8/16-bit and f64 buffer accesses.

## Tooling

Like the `integer` library, the feature is split between two flags, off by default in the libraries (the Luau command
line tools turn `Luau*` flags on by default):

- `LuauGpuKernels` (language): the parser accepts `@gpu`, and the `GpuKernel` lint runs
- `LuauGpuLibrary` (runtime): `luaL_openlibs` opens the `gpu` library
- with both, the type checker knows the `gpu` library

The `GpuKernel` lint reports `@gpu` functions that will run in the VM, with the reason and the location of the
unsupported code:

```
selftest.luau(223,15): GpuKernel: @gpu function 'usesTable' will run in the VM: tables are not supported on the GPU
```

Functions called by other `@gpu` functions are helpers; their unsupported code is reported through the kernels that
call them.

Since `luau` has the `gpu` library without a GPU backend, scripts written for `luau-gpu` also run there, in the VM.

## Embedding

The translator (`Luau.Gpu`) only depends on `Luau.Ast` and is always built. A host that wants GPU dispatches links
`Luau.Gpu.Vulkan`, enables both flags (or calls `luaopen_gpu`), and registers each chunk before running it:

```cpp
std::string error;
auto backend = Luau::Gpu::VulkanBackend::create(error); // nullptr if there is no usable device
backend->setFallbackHandler([](const std::string& message) { fprintf(stderr, "%s\n", message.c_str()); });
backend->load(chunkname, source.data(), source.size()); // chunkname as passed to luau_load
backend->install(L);
```

Any other backend can implement `lua_GpuBackend` (`lualib.h`) and install it with `lua_setgpubackend`.

## Layout

- `Ast`: the `@gpu` attribute
- `VM/src/lgpulib.cpp`: the `gpu` library, the VM implementation and the `lua_GpuBackend` hooks (`lualib.h`)
- `Gpu/src/GpuTranslate.cpp`: translates `@gpu` functions from the AST to GLSL compute shaders (`Luau.Gpu`)
- `Analysis`: `gpu` type definitions and the `GpuKernel` lint
- `Gpu/src/GpuVulkan.cpp`: Vulkan backend (`Luau.Gpu.Vulkan`); keeps a device copy of each buffer, uploads only buffers whose contents
  changed and downloads only buffers the kernel writes
- `Gpu/cli/Main.cpp`: `luau-gpu`, a host that runs a script with the backend and a small window library (SDL2)
- `Gpu/examples/selftest.luau`: runs kernels on the GPU and in the VM and compares the results
- `Gpu/examples/rasterizer.luau`: a software rasterizer drawing a 1M triangle model in real time
- `Gpu/src/GpuShaderCompiler.cpp`: SPIR-V compilation, validation and cross-compilation (`Luau.Gpu.Shaders`)
- `Gpu/bench/`: native Luau against GPU benchmarks, see [Gpu/bench/README.md](bench/README.md)
- `tests/GpuTranslate.test.cpp`: translator tests that don't need a GPU (part of `Luau.UnitTest`)
- `tests/GpuBackend.test.cpp`: shader compilation for every target, fallbacks, limits, threads (only with `LUAU_BUILD_GPU`)

## Building

The Vulkan backend, the shader compilers and `luau-gpu` are behind `LUAU_BUILD_GPU` (off by default) and need Vulkan,
shaderc, SPIRV-Tools, SPIRV-Cross, glslang and SDL2; the translator, lint and tests build everywhere, with CMake or the
Makefile. On NixOS, `Gpu/shell.nix` provides them:

```sh
nix-shell Gpu/shell.nix --run 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLUAU_BUILD_GPU=ON && ninja -C build luau-gpu'
./build/luau-gpu Gpu/examples/selftest.luau
./build/luau-gpu Gpu/examples/rasterizer.luau     # drag to rotate, wheel to zoom, G to switch GPU/VM
./build/luau-gpu --vm script.luau                  # run every kernel in the VM
```

`luau-gpu --emit=<glsl|spirv|msl-ios|msl-macos|gles> script.luau` prints the shaders of every kernel for a target
without running the script (see [Other platforms](#other-platforms)).

On an RTX 4080, the rasterizer draws 1M triangles at 1280x720 in about 5 ms per frame on the GPU, against about
760 ms per frame in the VM with native code generation. 99.8% of the pixels are identical between both. See
[Gpu/bench/README.md](bench/README.md) for more workloads, including the ones where the GPU doesn't win.

## Known limitations

- Every dispatch is synchronous and copies the buffers it writes back to the CPU; chaining dispatches without round
  trips needs data that stays on the GPU between dispatches.
- Kernels are translated from source when the host loads a chunk (`VulkanBackend::load`), and matched to compiled
  functions by chunk name and line; two `@gpu` functions starting on the same line can't be told apart and stay in
  the VM.
- Tools built on the upstream parser (luau-lsp, StyLua, selene) report `@gpu` as an invalid attribute.
