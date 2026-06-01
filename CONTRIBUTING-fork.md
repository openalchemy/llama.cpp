# Contributing to the OpenAlchemy fork

This file complements [`CONTRIBUTING.md`](CONTRIBUTING.md) (which is upstream
llama.cpp's policy) — it spells out what's specific to *this fork* on
`openalchemy/llama.cpp`.

## What belongs here vs upstream

| Change kind | Where it goes |
|---|---|
| A new ggml type, a CUDA kernel, a graph-build hook required for TurboQuant | **This fork** (`main` branch). Cherry-pick to upstream later when it stabilises. |
| Bug fix in any non-TurboQuant code (loader, sampler, server, fattn, etc.) | **Upstream** first. Then `git fetch origin master && git rebase origin/master`. |
| New model architecture, new quant type unrelated to TurboQuant | **Upstream** only. |
| Performance tweak that happens to help TurboQuant but is generic | **Upstream**. Doesn't need to live here. |
| Documentation about TurboQuant | **This fork** (`TURBOQUANT.md` + `docs/turboquant.md`). |

If you're not sure, default to upstream — keeping the diff vs upstream small
makes rebasing easier.

## Branch model

- `main` — production. Tracks upstream + carries the TurboQuant patch series.
- `<previous-version>-pre-rebase` — preserved before any upstream rebase, so
  there's always a working pre-rebase checkpoint to revert to.
- Topic branches — `feature/<short-name>` while iterating; merge into `main`
  via PR.

## Commit conventions

- `turboquant:` prefix for any commit touching TurboQuant types, kernels,
  graph hooks, or guards.
- `build:` for CMakeLists / packaging changes that affect how the fork is
  built (notably the `ENGINE_RT_WITH_TURBOQUANT` macro plumbing).
- Plain upstream-style prefixes for everything else (`server :`, `ggml :`,
  `cuda :`, etc.) — those should rarely be needed here; if you find yourself
  writing one, ask whether it belongs upstream instead.

Body should explain **why** in plain prose, **what** with file paths if the
change spans multiple files, and include numerical results (MSE, VRAM, t/s
delta) when relevant. The TurboQuant commit history is the closest thing to
a paper trail this work has — keep it readable.

## Tests you must run before opening a PR

```bash
# Build CPU-only first to catch the easy mistakes fast:
cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu --target test-turboquant -j

./build-cpu/bin/test-turboquant   # ← must pass; see expected MSE bounds inline

# Then CUDA, if your machine has it:
cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build-cuda --target test-backend-ops llama-app -j

./build-cuda/bin/test-backend-ops -o CPY   # ← all 248 cases pass, 0 fails

# E2E smoke (any head_dim=128 model):
./build-cuda/bin/llama.exe cli \
  -m <model>.gguf -ngl 99 -fa 1 -c 32768 \
  -ctk turbo3 -ctv turbo3 \
  -p "hi" -n 16
```

If `test-backend-ops -o CPY` regresses (any non-turbo CPY that previously
passed now fails), reject the change — the TurboQuant additions must be
strictly additive over upstream behaviour.

## Releasing a runtime pack

The runtime packs that `engine-desktop` consumes are built from
[`openalchemy/engine-runtime-cpp`](https://github.com/openalchemy/engine-runtime-cpp),
which vendors this fork as a submodule. Pack version numbers
(`0.3.0-beta.N`) are independent of llama.cpp commit shas — each pack
version pins a specific commit and is reproducible from that pair.

The convention for tagging fork commits that ship in a runtime pack:

```bash
git tag -a runtime-pack/0.3.0-beta.3 -m "engine_runtime 0.3.0-beta.3 shipped this commit"
git push openalchemy --tags
```

This lets anyone match a deployed pack version back to the source it was
built from, without having to dig through the runtime pack catalog manifest.

## License

Inherited from upstream llama.cpp: [MIT](LICENSE). All TurboQuant patches
in this fork are released under the same license.
