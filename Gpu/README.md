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
- `tests/GpuTranslate.test.cpp`: translator tests that don't need a GPU (part of `Luau.UnitTest`)

## Building

The Vulkan backend and `luau-gpu` are behind `LUAU_BUILD_GPU` (off by default) and need Vulkan, shaderc and SDL2;
the translator, lint and tests build everywhere, with CMake or the Makefile. On NixOS,
`Gpu/shell.nix` provides them:

```sh
nix-shell Gpu/shell.nix --run 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLUAU_BUILD_GPU=ON && ninja -C build luau-gpu'
./build/luau-gpu Gpu/examples/selftest.luau
./build/luau-gpu Gpu/examples/rasterizer.luau     # drag to rotate, wheel to zoom, G to switch GPU/VM
./build/luau-gpu --vm script.luau                  # run every kernel in the VM
```

`LUAU_GPU_DUMP=1` prints the generated GLSL of every kernel.

On an RTX 4080, the rasterizer draws 1M triangles at 1280x720 in about 5 ms per frame on the GPU, against about
2.9 s per frame in the VM with native code generation. 99.8% of the pixels are identical between both.

## Known limitations

- Every dispatch is synchronous and copies the buffers it writes back to the CPU; chaining dispatches without round
  trips needs data that stays on the GPU between dispatches.
- Kernels are translated from source when the host loads a chunk (`VulkanBackend::load`), and matched to compiled
  functions by chunk name and line; two `@gpu` functions starting on the same line can't be told apart and stay in
  the VM.
- Tools built on the upstream parser (luau-lsp, StyLua, selene) report `@gpu` as an invalid attribute.
