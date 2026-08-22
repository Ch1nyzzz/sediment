# 系统设计：4×GPU 上的流式门控 TTT（Qwen3-4B）

结论：**放弃 verl colocate，改为 3 卡常驻推理 + 1 卡常驻训练的窗口化流水线。**
verl 的相位切换（sleep/wake + FSDP）为大 batch 离线 RL 设计；流式方法每窗口只有
几十秒的 LoRA 蒸馏量，且 predict-then-update 要求服务永不中断。
代码基础是 P0（VllmEngine 打分、experience 构块、spans 对齐）而非 resid_verl；
从后者只搬 credit_belief 的 loss 逻辑与 hindsight 构块函数（去 verl 依赖）。

## 角色划分

```
GPU 0-2  vLLM 常驻服务（TP=1，独立引擎 + 任务级路由）
         rollout / hindsight 打分 / 门控验证 / 冻结快照
GPU 3    trainer 常驻（HF + peft，bf16，flash-attn2）
         窗口蒸馏 → 候选 adapter；空闲回填 HF 打分
CPU      环境进程池、调度器、adapter registry、experience buffer
```

三个关键机制：

1. **multi-LoRA 单引擎**：`--enable-lora --max-lora-rank 32 --max-loras 8`；
   session 历代版本、候选、冻结快照共存，按请求选择。门控 A/B 行为验证
   = 两个 adapter id 的并发请求，无需切换。
2. **runtime 热载**：trainer 产出 adapter 入 registry（目录+版本软链），
   引擎走 `/v1/load_lora_adapter`，永不重启（aTTT 验证 <100ms）。
3. **prefix caching + episode 粘性路由**：多轮 episode 每步重 prefill 增长的对话，
   APC 使其增量化；同一 episode 必须哈希路由到同一引擎，否则缓存命中归零。
   这是最大的单项利用率杠杆。

## 窗口化流水线（G=1 修订版）

窗口 W=16–32 个任务并发（并发度靠任务数，不靠组内采样）：

```
时刻 k:   GPU0-2  窗口 k rollout+打分（adapter v(k-1)）
          GPU3    训练窗口 k-1 蒸馏 → 候选 v(k)'
窗口尾:   GPU0-2  候选 v(k)' 门控验证（重放+探针，与 rollout 混批）
          通过 → merge 发布 v(k) → 热载 → 窗口 k+1
```

任何窗口的作答权重不含本窗口数据（陈旧度恒 1 窗口）。
流式主循环**不做 teacher 采样**：teacher 只是带 E_x 的 2 次 prefill 打分
（hindsight.py 的"只多一次 prefill"设计）。

## 显存账（Qwen3-4B：权重 bf16 ≈ 8GB；KV ≈ 144KB/token，36 层 × 8 KV 头 × 128 dim）

| 卡型 | 单卡 KV 预算 | 12K 上下文可养并发 | 建议 |
|---|---|---|---|
| 24GB | ~10GB ≈ 70K tok | 5–6 条满长（APC 后 10+） | 开 fp8 KV；W=12 |
| 40GB | ~24GB ≈ 165K tok | ~14 条 | W=16–24 |
| 80GB | ~60GB ≈ 400K+ tok | 不构成约束 | W=32 |

trainer 卡：8GB 权重 + LoRA 可忽略 + 12K 序列激活；40GB+ 免重计算 micro-batch 2–4；
24GB 开 gradient checkpointing。**不在 trainer 卡起 vLLM**（显存打架）。

吞吐估算（A100 级，待实测校准）：G=1 后每任务生成 ~6K tok；窗口 16 任务纯解码
<1 min，加环境与打分 ~1–3 min/窗口。trainer 每窗口只训高 |w_t| span
（只写残差是方法本身），15–30s，躲在 rollout 阴影里。
300 任务流 ≈ 1–2 小时；P1+P3 全程（3 种子 × 3 流序）≈ 3–5 GPU 天。

## 利用率杀手（按杀伤力）

1. 逐任务串行（batch=1 解码）——窗口化是前提。
2. 不开 APC / 路由不粘性——prefill 成本线性变平方。
3. 重启引擎换 adapter——必须 runtime 热载 + 版本化命名。
4. 环境执行阻塞事件循环——EnvScaler 工具是进程内 exec，进程池 + asyncio；
   并发前加基本隔离（README 已注明无沙箱）。
5. 打分挤占 rollout——with/without-E 前缀不同无法互享缓存；窗口尾集中批处理，
   trainer 空闲分流。
6. CPU 侧：spans 对齐与分词开 worker 池；jsonl 异步落盘。
7. 快照评测=只读 adapter id 的低优先级请求，跑在窗口间隙。
