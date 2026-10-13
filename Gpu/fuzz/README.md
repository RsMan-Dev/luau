# Differential fuzzing of @gpu kernels

`difffuzz.py` generates random kernels in the GPU subset, runs each of them on the GPU and in the VM through
`luau-gpu`, and reports every output that differs.

```sh
nix-shell Gpu/shell.nix --run 'python3 Gpu/fuzz/difffuzz.py --luau-gpu build/luau-gpu --rounds 10 --family all --profile tame'
```

Options: `--seed` (rounds use consecutive seeds, so a failure is reproducible), `--rounds`, `--kernels` per round,
`--family int|float|vector|both|all`, `--profile tame|extreme`, `--show N` failing kernels printed,
`--show-precision` and `--show-fallbacks N` to look at the cases that aren't counted as failures, `--timeout` for the
whole campaign. The script of a failing campaign is kept (`--workdir`) and can be run directly with `luau-gpu`; giving
it `round:kernel` as its argument skips the cases before that one.

## How a campaign runs

A whole campaign is one script run by one `luau-gpu` process, with one Vulkan device, the way a game would use it:
each round's kernels live in their own `do` block, and a `check` function runs every kernel on the GPU, then in the VM,
and prints the words that differ. Generated loops count their iterations against a `guard` that breaks out after 4096
of them, identically on both sides, so no kernel can keep the GPU or the VM busy for long and no process ever needs to
be killed for taking too long.

The output is line buffered: when the process dies (a shader compiler bug can take it down), the case after the last
one printed is reported with the signal, and the campaign resumes after it in a new process.

## Families

| Family | Kernels | Comparison |
|---|---|---|
| `int` | integer sources, literals within 32 bits, `+ - * // %`, `bit32`, `math` rounding, min/max/clamp, helpers, loops | bit-exact |
| `float` | adds floats, `/`, `^`, `sqrt`, `fmod`, float literals including values beyond 32 bits | f32 outputs: relative 1e-4; integer outputs derived from floats are reported as "precision" |
| `vector` | vector arithmetic, dot, cross, normalize, magnitude, floor/ceil, min/max | relative 1e-4; differences after a division, rounding, comparison or normalization are "precision" |

In every family, an f32 output that is NaN on one side only is an exact mismatch: rounding never explains it.

## Profiles

- `extreme` feeds values at the edges of the int and uint ranges: most dispatches overflow 32 bits and must be
  discarded and rerun in the VM, which checks that no wrong value ever leaves the GPU.
- `tame` keeps values small so that more kernels stay on the GPU, which checks the GPU arithmetic itself; a fallback
  there is worth a look (`--show-fallbacks`). Many still come from `bit32` results of negative numbers, which are large
  unsigned values that later overflow.

## What counts as a difference

Integer values on the GPU are exact or the dispatch reruns in the VM, so integer outputs of integer computations must
match bit for bit. Floats are f32 on the GPU and doubles in the VM; the accepted differences are:

- f32 rounding, including integers above 2^24 held in floats and everything derived from them
- GPU division, `sqrt` and transcendental functions, which Vulkan doesn't require to be correctly rounded (`pow` can be
  off by about 1e-4 relative); `floor`, `ceil` and comparisons can turn an ulp into a whole unit or another branch
- NaN bit patterns (the VM's depend on the CPU)
- the sign of an integer zero: `-x` for an integer `x == 0` is `-0` in the VM and `0` on the GPU
