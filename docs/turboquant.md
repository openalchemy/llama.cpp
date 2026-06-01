# TurboQuant KV Cache — engine-runtime-cpp 集成方案

**Status:** Draft
**Date:** 2026-05-30
**Scope:** `engine-runtime-cpp/` (vendored llama.cpp fork on branch `oa-turboquant`, new CUDA quant kernels, KV cache + Flash Attention 改动)、`engine/` (Go agent runtime ABI 增加 KV quant 选项 + UI 透传)、`engine-desktop/` (Settings → Runtime 增加 KV 量化开关)。**不**改 avionics、grid、scheduler。

---

## 1. Background

### 1.1 当前现状

- engine-runtime-cpp 通过 `third_party/llama.cpp` git submodule 引入 llama.cpp，pinned 在 commit `b28a2f372a4a470a90ad10f93654e5dc33e78949` (upstream `ggml-org/llama.cpp`)。
- KV cache 当前以 FP16 / Q8_0 / Q4_0 三档存储 (`-ctk` / `-ctv` flags)。Q4_0 已经在长上下文场景下损失明显（HumanEval、MMLU 在 16k+ context 下 ~2-4 个百分点）。
- RTX 5080（16 GB VRAM）跑 7B-class 模型加载完权重后，KV cache 在 ~8k context 已经占 2-3 GB，限制了 batch / context 上限。
- 我们的 runtime pack 是 NVIDIA CUDA-only（dynamic_runtime + ggml-cuda backend），所以 GPU 路径必须是 CUDA kernel，不能用 ROCm 路径。

### 1.2 Why TurboQuant

TurboQuant (Zandieh et al., Google Research, ICLR 2026, arXiv 2504.19874) 是 **KV cache** 量化方法（不动权重），核心思路：

1. **随机正交旋转**：每层 K / V 向量左乘一个随机正交矩阵 Π ∈ ℝ^(d×d) (Haar 分布)；高维下旋转后每个坐标分布趋近一个集中的 Beta 分布。
2. **逐坐标标量量化**：因为旋转后坐标近似 i.i.d.，对每维独立做 Lloyd-Max 最优标量量化即近最优。
3. **反量化**：用相同 seed 重生成 Π，反查码本得到 ỹ，再 Πᵀ·ỹ 旋回原空间。

**关键属性**：
- *Data-oblivious*：无需校准数据集、无需训练、无 per-model codebook。Seed 即码本规约的全部"模型"。
- *Bit-width*：3 bit / 坐标做到与 FP16 等价（长上下文 LLM 上 paper Table 5），2.5 bit 边际可接受退化。
- *已有 OSS 参考实现*：`Pascal-SAPUI5/llama.cpp-turboquant` (ROCm-only), `vivekvar-dl/turboquant` (Triton/vLLM), `yashkc2025/turboquant` (Python ref)。

### 1.3 期望收益（RTX 5080 / Llama 3.1 8B 估算）

| 场景 | Baseline FP16 KV | TurboQuant 3-bit KV | 收益 |
|---|---|---|---|
| 8k context | 1.0 GB | ~0.19 GB | -81 % |
| 32k context | 4.0 GB | ~0.75 GB | 节省 3.25 GB → 16k → 64k 同卡可达 |
| 同卡最大 batch (8k ctx) | 4 并发 | 12-16 并发 | 3× 吞吐 |

### 1.4 工程上的简化

直接实现 paper 的 Haar 正交矩阵 (QR(随机正态矩阵)) 在 GPU 上代价过高。参考开源 fork 用 **Fast Walsh-Hadamard Transform (FWHT)** 替代——FWHT 是 ±1 项的正交矩阵，旋转后近似 Beta 分布的性质仍成立，但 kernel 只需 O(d log d) 加减法，无浮点乘。我们采用同样近似。**Tradeoff：** 失去对 paper 理论保证的严格性，换 ~10× kernel 速度；准确性损失 paper 作者本人 issue 区回复"在 head_dim ≥ 64 时不可观测"。

---

## 2. Architecture

### 2.1 改动栈层

```
engine-desktop UI (Settings → Runtime)
    │  kv_cache_quant: "fp16" | "q8_0" | "turbo3" | "turbo2"
    ▼
engine-agent (Go) — load_params extra_args_json
    │  {"cache_type_k": "turbo3", "cache_type_v": "turbo3"}
    ▼
engine_runtime ABI (C, dynamic_runtime pack)
    │  engine_rt_load_params_v1.extraArgsJSON 透传
    ▼
llama.cpp fork (branch oa-turboquant)
    │
    ├── ggml/src/ggml-quants.{c,h}      新增 block_turbo3 / block_turbo2 + CPU ref
    ├── ggml/src/ggml-cuda/
    │   ├── turboquant.cu               新 kernel: FWHT + scalar-quant encode/decode
    │   ├── fattn-tile.cu               Flash Attention 内联解 K / V
    │   └── cpy.cu                      cpy(F16→TURBOx) 路径
    ├── src/llama-kv-cache.cpp          head_dim==128 校验 + 分配大小算
    └── src/llama-model.cpp             cache_type_k/_v 解析增加新枚举
```

### 2.2 数据流（生成阶段，每 token 注入 K/V 时）

```
                ┌─────────────────────────────────────┐
attention proj  │  Q_t, K_t, V_t  (FP16, head_dim=128) │
                └─────────────────────────────────────┘
                                │
                  (1) 仅 K, V 进入 turbo 路径；Q 走 FP16
                                │
                                ▼
                ┌─────────────────────────────────────┐
                │  FWHT(K_t), FWHT(V_t)               │  per-head; in-place; layer-seeded
                └─────────────────────────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────────┐
                │  L2 norm‖K_t‖, ‖V_t‖ stored FP16   │
                │  scalar quantize each coord → 3 bit │
                └─────────────────────────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────────┐
                │  block_turbo3 (head_dim=128 →       │
                │     48 bytes payload + 2 bytes norm)│
                └─────────────────────────────────────┘
                                │
                                ▼
                    cache slot[layer, head, t]
```

### 2.3 数据流（attention 计算，每 query token）

Flash Attention tile loop 内逐块 dequantize：

```
for each KV-tile of size BlockN:
    1) load block_turbo3 → registers
    2) dequant: scale = norm; y = codebook[idx];
       (FWHT 逆变换合并进 Q·Kᵀ 之前——把 Q 乘上 H 而非把 K 乘上 Hᵀ，
        因为 Q 一次 vs K 多次，节省 d log d 次操作的 N_kv 倍)
    3) 用解出的 K_tile 算 attention scores
    4) softmax(scores)
    5) 同样 dequant V_tile，加权求和
```

**关键工程决策**：FWHT 是自逆 (H·Hᵀ = d·I)，所以反旋等价于再做一次 FWHT 然后除以 d。把 Q 侧旋转一次（每 query token 只做一次）替代 K 侧每个 cache slot 反旋一次——计算量从 O(N_kv · d log d) 降到 O(d log d)，N_kv 越大节省越多。

### 2.4 与现有 ABI 关系

不动 ABI 二进制布局。`engine_rt_load_params_v1.extraArgsJSON` 已经存在（runtime_dynamic.go:101），用它透传 `{"cache_type_k": "turbo3", "cache_type_v": "turbo3"}` 到 llama.cpp 的 `llama_context_params.type_k/_v`。avionics、scheduler、NATS 协议**零改动**。

---

## 3. Schema / Code changes

### 3.1 新增 ggml 量化类型

`ggml/include/ggml.h` (enum `ggml_type`):

```c
GGML_TYPE_TURBO3 = 41,  // 3-bit TurboQuant; per-block = head_dim coords + 1 FP16 norm
GGML_TYPE_TURBO2 = 42,  // 2-bit TurboQuant; same layout, smaller payload
```

块定义 (`ggml/src/ggml-quants.h`)：

```c
#define QK_TURBO3 128   // head_dim 必须等于 QK_TURBO

typedef struct {
    ggml_fp16_t norm;            // L2 norm of post-FWHT block, kept FP16
    uint8_t     qs[48];          // 128 coords × 3 bit = 48 bytes
} block_turbo3;
static_assert(sizeof(block_turbo3) == 50, "block_turbo3 must be 50 bytes");

typedef struct {
    ggml_fp16_t norm;
    uint8_t     qs[32];          // 128 × 2 bit = 32 bytes
} block_turbo2;
static_assert(sizeof(block_turbo2) == 34, "block_turbo2 must be 34 bytes");
```

> **为何把 norm 放块内而不是块外 FP32：** 块内 FP16 已足，避免 KV 内存里穿插两条 stride。FP16 与 FP8 都试过，FP16 在 ablation 上稳；FP8 在小模型早期 token 偶发溢出。

### 3.2 码本（precomputed Lloyd-Max）

`ggml/src/ggml-cuda/turboquant_codebook.h` —— 由 `tools/turboquant/gen_codebook.py` 离线一次性算出（Beta(d/2, d/2) 分布上跑 Lloyd-Max 收敛），编进库里作 `__constant__` 常量：

```c
// d = 128, Lloyd-Max optimal centroids over Beta(64, 64) on [-1, 1].
__constant__ float TURBO3_CODEBOOK[8] = { -0.7536f, ..., +0.7536f };
__constant__ float TURBO2_CODEBOOK[4] = { -0.8233f, ..., +0.8233f };
```

### 3.3 llama.cpp 入口

`src/llama-model.cpp`：`llama_model_kv_override_type_to_ggml_type` 解析增加 `"turbo3" → GGML_TYPE_TURBO3` 等。

`src/llama-kv-cache.cpp`：在分配 KV buffer 时若 type_k 或 type_v 为 TURBO*，校验 `model.hparams.n_embd_head_v == 128`，否则返回错误并回退 FP16。

`include/llama.h`：`enum llama_kv_cache_type` 不需要变（已经 1:1 对应 ggml_type）。CLI flag `-ctk turbo3` 自动可用。

### 3.4 engine-runtime ABI 不变

`engine_rt_load_params_v1` 已有 `extraArgsJSON` 字段，engine-runtime-cpp 的 `engine_rt_load_model` 内部解析这个 JSON 把 `cache_type_k/v` 塞进 `llama_context_params`。**新增** UI 友好的字符串枚举 `"fp16" | "q8_0" | "turbo3" | "turbo2"`，把映射放在 runtime 侧不放在 agent 侧（避免 agent 知道 llama.cpp 的具体枚举值）。

### 3.5 engine agent 透传

`engine/internal/agent/runtime.go`：模型加载请求里 `KVCacheTypeK` / `KVCacheTypeV` 两个 string 字段，序列化进 ExtraArgsJSON。默认值空 → 走 llama.cpp 默认（FP16）。**不**默认开 turbo3——必须显式 opt-in，避免老模型因为 head_dim != 128 加载失败。

### 3.6 engine-desktop UI

`engine-desktop/src/routes/Settings.tsx` → Runtime 区块新增"KV Cache Quantization"下拉：

| 选项 | 含义 | 推荐场景 |
|---|---|---|
| FP16 (default) | 不动 | 短上下文，最高质量 |
| Q8_0 | llama.cpp 现有 8 bit | 平衡 |
| Q4_0 | llama.cpp 现有 4 bit | 旧路径，存档 |
| **TurboQuant 3-bit** | 新 | 长上下文 / 大 batch |
| **TurboQuant 2-bit** | 新（实验） | 极限压缩 |

选择变更需要重载模型——和现有 Quant Strategy 下拉同样的 UX。head_dim != 128 的模型在 picker 里把 turbo* 灰掉 + 显示 tooltip "Not supported by this model"。

---

## 4. Implementation phases

### Phase 0 — Fork setup (0.5 d) — **Already done locally**

1. Clone 了顶层 workspace 副本 `openalchemy/llama.cpp` (185 MB, depth 200) — 作为独立开发树，不在 submodule 里。
2. Base 取 upstream `192d8ae` "CUDA: missing PDL sync for FWHT, better fallback (#23690)" —— **upstream 已经合了 FWHT CUDA kernel** (`ggml/src/ggml-cuda/fwht.cu`, `c1f1e28d`)，TurboQuant 旋转直接复用，Phase 2 工作量减少约 1.5 天。Vulkan 路径也有 FWHT (`48e7078`) 留作 future。
3. Branch `oa-turboquant` 已创建并 checkout，本地存在，未 push。
4. Push 给 `github.com/openalchemy/llama.cpp` + 改 `engine-runtime-cpp/.gitmodules` url + branch 字段，是 Phase 4 收尾时操作（早 push 反而锁死 base，开发期保持本地）。

**Verify:** `openalchemy/llama.cpp` 顶层目录存在，`git branch --show-current` 返回 `oa-turboquant`，HEAD = `192d8ae`。

### Phase 1 — CPU 参考实现 + 单元测试 (1.5 d)

文件：`ggml/src/ggml-quants.c` (encode/decode for TURBO3/TURBO2), `tests/test-turboquant.cpp`。

```c
void quantize_row_turbo3(const float *src, block_turbo3 *dst, int64_t k);
void dequantize_row_turbo3(const block_turbo3 *src, float *dst, int64_t k);
```

测试：
- Round-trip MSE 在 paper 的预期范围内（Beta(64,64) 下 3-bit ≤ 0.018）
- d=128 强制；其他 d 返回错误
- Codebook 加载正确
- FWHT 自逆性（H·H·x / d == x）

**Verify:** `ctest -R turboquant` 全过。性能不重要，CPU 是 fallback 路径。

### Phase 2 — CUDA encode/decode kernels (3 d)

文件：`ggml/src/ggml-cuda/turboquant.cu`, `turboquant.cuh`。

Kernels：

1. **`fwht_inplace_d128<<<...>>>`** —— shared-memory FWHT (7 stages, 128-thread block, 1 head/block)。Spec: 32 warps × 1 head, ~0.6 µs/head on RTX 5080。
2. **`encode_turbo3<<<...>>>`** —— 接受 post-FWHT vector，算 L2 norm，归一化到 [-1, 1]，对每坐标二分查找最近码本 idx，pack 3 bit 进 48 字节。
3. **`decode_turbo3<<<...>>>`** —— bulk dequantize 一整个 cache row 到 FP16 scratch（给 prefill 路径用，不给 decode 路径用——decode 用 fused-in-attention 路径见 Phase 3）。

集成到 ggml-cuda dispatcher：`ggml_cuda_cpy_f16_turbo3` 走 `cpy.cu`，attention 写回路径走 `set-rows.cu`。

**Verify:** CUDA round-trip MSE 与 CPU 一致（容差 1e-4，rounding 差）。`llama-bench -m llama-3.1-8b -ctk turbo3 -ctv turbo3 -fa 1 -p 4096 -n 128` 完成无 NaN。

### Phase 3 — Flash Attention 集成 (2 d)

文件：`ggml/src/ggml-cuda/fattn-tile.cu`（流式 dequant 进 attention tile）, `fattn.cu` (dispatch)。

关键改动：
- 在 `fattn-tile.cu` 加 `template<ggml_type type_K>` 分支；type_K=TURBO3 时 tile load 先 dequant 进 shared mem。
- Q 侧旋转：在 attention entry 处对 Q 做一次 FWHT；K 反旋免去（H 自逆性 + 系数 1/d 合并到 score scaling）。
- 把 1/d 系数合并进 `scale_softmax` 已有的 1/√d。

**Verify:**
1. `tests/test-backend-ops -o ATTN -t TURBO3` 与 FP16 baseline cosine sim ≥ 0.999
2. perplexity wikitext-2 (Llama 3.1 8B, 4k ctx): FP16 baseline vs turbo3 差 ≤ 0.05
3. `llama-bench` 实测 8k ctx KV 占用从 1.0 GB → 0.19 GB，token/s 退化 ≤ 5 %

### Phase 4 — engine-runtime-cpp + engine agent 串联 (1 d)

1. `engine-runtime-cpp/src/engine_runtime.cpp`: `engine_rt_load_model` 解析 `extraArgsJSON.cache_type_k/v`，map "turbo3" → `GGML_TYPE_TURBO3` 塞进 `llama_context_params`。
2. `engine/internal/agent/runtime.go`: 新 `KVCacheTypeK` / `KVCacheTypeV` 字段，UI 透传。
3. `engine/internal/agent/local.go`: `/local/status` payload 增加 `kv_cache_quant` 字符串（已加载值，给 UI 反馈）。

**Verify:** 从 engine-desktop 选 turbo3 → 加载 Llama 3.1 8B → status 显示 `kv_cache_quant: "turbo3"` → 跑一次 chat 拿到正常响应。

### Phase 5 — engine-desktop UI (0.5 d)

`src/routes/Settings.tsx`: KV Cache Quant 下拉 + 解释 tooltip + 已加载模型的 head_dim 检查 + i18n key `settings.runtime.kv_quant.*`。

**Verify:** UI 切换 → 重载模型 → bottom bar 显示新的 kv quant 标记（沿用 capability pill 体系）。

### Phase 6 — Docs (0.25 d)

1. `engine-runtime-cpp/README.md` 增加"KV cache quantization"节，列支持矩阵。
2. `engine-desktop/RELEASE_NOTES.md` for 0.3.x 包含 TurboQuant 段落（说明默认 off，beta runtime pack 已可装）。
3. 一个 Loom / 截图：8k ctx 同一模型 FP16 vs turbo3 VRAM 占用对比。

### Phase 7 — Runtime pack release（多版本并存，单 channel）(1 d)

**目标：** 通过现有 runtime pack 渠道把 TurboQuant llama.cpp 作为新版本推给用户，**和 stable 的 0.2.0 并存**，操作员在 Settings → Runtime 里一键切换、A/B 比较。**不新建 beta channel**，沿用现有的 `stable.json` manifest，扩展成多版本广播。

#### 7.1 命名 + 版本策略

| 字段 | 现有 | 新增 |
|---|---|---|
| `pack_id` | `llamacpp-cuda12-windows-x86_64` | **同 id** |
| `version` | `0.2.0` | `0.3.0-beta.1` （semver pre-release tag，自动排在 0.2.0 之后，但 auto-update 默认跳过 pre-release） |
| 安装路径 | `%LOCALAPPDATA%\OpenAlchemy\runtimes\llamacpp-cuda12-windows-x86_64\0.2.0\` | `…\0.3.0-beta.1\` （同 pack_id 不同 version 子目录，磁盘共存） |
| Update manifest | `updates.openalchemy.io/runtime/llamacpp-cuda12-windows-x86_64/stable.json` | **同一个文件**，扩展 schema |

> **为什么同 pack_id 同 channel：** 选择切换是 desktop UI 范畴的事，不是网络分发层的事。维护一个 manifest 文件、一份签名、一条 Discord 公告，比维护 stable/beta 两条平行链路省心。pre-release 后缀 `-beta.1` 本身就是 "不要自动升过去" 的语义信号；auto-update 路径默认锁 stable major.minor.patch 不跨 prerelease，要装 beta 是 UI 显式行为。

#### 7.2 Manifest schema 扩展

现 `stable.json` 单版本：

```json
{ "version": "0.2.0", "url": "...", "signature": "...", "notes": "..." }
```

扩展为多版本（向后兼容，老 agent 只读顶层字段）：

```json
{
  "version":       "0.2.0",
  "url":           ".../downloads/0.2.0.zip",
  "signature":     "...",
  "notes":         "...",
  "available": [
    {
      "version":   "0.2.0",
      "url":       ".../downloads/0.2.0.zip",
      "signature": "...",
      "notes":     "...",
      "channel":   "stable",
      "default":   true
    },
    {
      "version":   "0.3.0-beta.1",
      "url":       ".../downloads/0.3.0-beta.1.zip",
      "signature": "...",
      "notes":     "TurboQuant KV cache beta — opt-in via Settings → Runtime",
      "channel":   "preview"
    }
  ]
}
```

老的 `engine-desktop` 0.2.x 只看顶层 `version/url/signature` → 仍升级到 0.2.0、看不到 beta，零回归。新的 desktop（同步发的 0.3.0）看 `available[]`，把每个条目列出来供切换。

> **`channel` 字段在条目级，不是 manifest 级。** 这里只是给 UI 一个 tag 显示用（"Stable" / "Preview"），不是 update 路由信号。`default: true` 指明 auto-update 跟踪的目标。

#### 7.3 selections.json 改动

无 schema 变更。`engine/internal/runtime/selection.go` 的 `Selections` 结构不动。仅 desktop UI 改写它指向不同 version：

```json
{ "llama": { "pack_id": "llamacpp-cuda12-windows-x86_64", "version": "0.3.0-beta.1" } }
```

agent 收到的还是 pack_id@version，完全不感知 stable / preview 概念。

#### 7.4 UI 改动 (engine-desktop)

Settings → Runtime 改成版本列表 + tag：

```
Runtime: llamacpp-cuda12-windows-x86_64                Auto-update [✓]

  ✓ 0.2.0          (Active · Stable · default)
    0.3.0-beta.1   (Available · Preview · TurboQuant)   [Install] [Switch to this]
```

切换不删除其他版本（保 rollback）；仅 rewrite `selections.json` 的 `llama.version`。Switch 触发 `runtime_packs.rs:reload_runtime()` 给 agent 发 `engine_rt_shutdown` + `engine_rt_init` 重载。Auto-update 只看 `default: true` 那条，preview 版本永远靠用户手动 Install。

#### 7.5 测试切换流程

操作员的 A/B：

1. 0.2.0 上同一模型加载，记 chat tokens/s + KV VRAM (`/local/status` 已有字段)。
2. Switch → 0.3.0-beta.1，**同样模型同样设置**重载，再记一次。
3. （可选 / 不进首版）Settings → Runtime 上 "Compare last 10 jobs" 链接，弹 modal 展示两轮 metrics 平均。

#### 7.6 发布操作（mirror release skill 现有流程）

复用 [[reference_oci_updates]] 上 nginx vhost，文件树仅新增 `0.3.0-beta.1.zip(.minisig)`：

```
/var/www/updates/runtime/llamacpp-cuda12-windows-x86_64/
  stable.json                       # 改 schema：加 available[]
  downloads/
    0.2.0.zip                       # 现有
    0.2.0.zip.minisig
    0.3.0-beta.1.zip                # NEW
    0.3.0-beta.1.zip.minisig
```

签名走同一个 minisign 2635A8BB key（[[reference_signing_keys]]）。Discord 公告：发到现有 `#announcement` 加 `[BETA]` / `[PREVIEW]` 前缀；不开新频道。

**Phase 7 Verify:**
- 老 desktop (0.2.x) 看 stable.json 顶层字段：仍是 0.2.0，零变化，看不到 beta
- 新 desktop (0.3.0) Settings → Runtime 看到两条：0.2.0 (Active · default) + 0.3.0-beta.1 (Preview)
- Install 0.3.0-beta.1 → Switch → 重载同模型 → chat 工作 → KV VRAM 下降
- 回切 0.2.0 → 重载 → chat 仍工作（rollback 路径）
- Auto-update 不会偷偷把 0.3.0-beta.1 设为 active（只跟踪 `default: true`）

**总计：9.75 个工作日**（Phase 0 已完成本地准备 -0.5d；Phase 2 因 upstream FWHT -1.5d；Phase 7 +1d；Phase 6 -0.25d。净减 0.25d 还有意外 buffer）。

---

## 5. Risks and mitigations

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| head_dim != 128 模型不支持 | 高 | 中 | UI 检测 + 灰选项；fallback 自动 FP16；spec §3.3 已校验 |
| FWHT 近似偏离 paper 理论 | 中 | 中 | Phase 3 用 wikitext + HumanEval 实测；perplexity 退化 > 0.1 则回退 Haar |
| CUDA kernel 与上游 llama.cpp rebase 冲突 | 中 | 中 | 改动隔离在 `ggml-cuda/turboquant.*`；fattn-tile 改动 ≤ 50 行 + 用 `template<ggml_type>` 分支避免 if-else 扩散 |
| 2-bit 模式准确性不达标 | 高 | 低 | 标记 "experimental"；不默认开；UI tooltip 说明 |
| 与 grouped-query attention (GQA) 交互 bug | 中 | 高 | Phase 3 test matrix 必须包含 Llama 3.1 8B (GQA 8 head 1 kv-head) + Qwen2 7B (GQA 28:4) |
| Old GGUF 模型 cache type 配合 turbo 写入失败 | 低 | 低 | engine-runtime-cpp 层做 type compatibility check，UI 之前就拒绝 |
| upstream llama.cpp 移除某 ggml_type slot 41/42 | 低 | 中 | 用 ≥ 41 起手避开当前已用编号；rebase 检查 |
| Multi-tenant 误配置：一个 worker 跑多模型时 KV 选项混淆 | 中 | 低 | KV quant 是 per-model-load 参数（不是 worker-global），supervisor slot 字典自然隔离；UI 在 model picker 里展示当前选项 |

### 5.1 多租户安全性

KV cache 是 worker 本机内存，不跨 worker / 不跨租户。TurboQuant 不引入任何 cross-tenant 数据共享。Seed 由 layer index 派生（固定 `seed = layer * 0x9E3779B97F4A7C15`），不依赖任何租户 / 模型 / 域 标识符。审计点：

- ✅ avionics 不感知 KV quant 类型（只收模型 ID + token 用量）
- ✅ NATS 协议不变
- ✅ scheduler 不感知（worker 的 KV quant 不进 capability advertise）
- ✅ 跨 worker 漂移：不同 worker 可以用不同 KV quant 跑同一模型，结果仍数值一致（同样 FWHT seed），租户在 grid 看不出差别

---

## 6. Out of scope

- **权重量化**：不动 GGUF 的 Q4_K_M / Q5_K_M 等现有体系。TurboQuant 仅 KV cache。
- **训练 / fine-tune 路径**：本 spec 只涵盖推理。
- **CPU 性能优化**：CPU 路径仅做正确性参考，不做 AVX-512 调优。
- **MoE expert weight rotation**：paper 中的 expert 路径压缩在 OSS 路径之外，超出本期范围。
- **ROCm / Metal**：我们 runtime pack 是 CUDA-only；ROCm/Metal 后续 phase。
- **Embedding / Rerank 模型**：embed 模型不用 KV cache，rerank 用极短 KV，本 spec 不覆盖。
- **跨 runtime pack 兼容**：本期只动 cuda12 pack；CPU pack 跟着 ggml-quants.c CPU ref 自动可用但不上 UI。
- **avionics / grid / scheduler 改动**：无。
- **Discord / 公告自动化**：标准 release 流程，不在本 spec 里另开。

---

## 7. References

- Paper: Zandieh, Daliri, Hadian, Mirrokni — *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* — arXiv 2504.19874, ICLR 2026
- Reference impls (调研对照):
  - `Pascal-SAPUI5/llama.cpp-turboquant` — llama.cpp fork (ROCm only), 我们 CUDA 路径的结构参考
  - `vivekvar-dl/turboquant` — Triton kernels + vLLM 集成
  - `yashkc2025/turboquant` — Python 参考（用作 CPU 测试 ground truth）
- 我们 fork 起点：upstream commit `192d8ae` "CUDA: missing PDL sync for FWHT, better fallback (#23690)" — 后于 engine-runtime-cpp 当前 pin `b28a2f3`，包含 FWHT CUDA kernel。Fork tree 在 `C:\Users\Kazuki\openalchemy\llama.cpp` (顶层 workspace, 非 submodule), branch `oa-turboquant`
- 相关本仓 spec:
  - `2026-04-22-runtime-extension-packs.md` — runtime pack 架构，本 spec 改动落在 pack 内部
  - `2026-05-25-engine-worker-registration.md` — KV quant 不影响 worker capability advertise
