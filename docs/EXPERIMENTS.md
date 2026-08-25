# 实验计划（P1–P4）

原则：每个阶段证一个可证伪命题，P1 是 go/no-go。P0（信号存在性探针）已在 resid 完成：
100 任务 mean gain +0.096、dead-zone 分类、belief 通道 w_t 测量——作为论文 Fig 1，
与部署协议（G=1）分属两章。

## P1 任务内内化（go/no-go）

**命题**：belief 加权蒸馏能把 E_x 的 in-context 增益写进权重。

逐任务、用完重置。每任务：一条 student 轨迹 + 检索/自身结果构造 E_x →
蒸馏进 LoRA（r=32 挂 Q/V+MLP，1–2 epoch，lr 5e-5）→ 不带 E_x 重试。

对照：base 不适应 / E_x 留 context（ICL 上界）/ 无加权全 token 蒸馏 /
自 NTP（aTTT Self 信号，最重要基线）/ 仅成功轨迹 SFT。

判据：≥3 种子下恢复 **ICL 增益的 ≥50%**（≥+0.05）且显著优于无加权与自 NTP。
不达标先修方法（加步数 / 9B 底座 / on-policy 蒸馏），不进 P3。

附加测量：①内化后把 E_x 放回 context 再测（NCA 上下文回归检查）；
②任务池按 P0 dead-zone 取 mixed + rescued（all-success 无信号，all-fail 无底力）。

### P1 实证快照（2026-08-23，tail-200 池，Qwen3-4B，G=1 五臂）

- **v0（仅结局块）被证伪**：ICL 配对净 −3.5pp（救6/坏13），首答成功任务留存被打到 0.68
  （重采样基线 0.90）——贫瘠证据造成净伤害；uniform 全 token 蒸馏在同任务重测下靠
  背题拿 100% 留存（协议伪影 → 信念掩码后消失）。
- **v1（steps 块：动作逐字+每步 ok/ERROR+真结算）证实证据完整度假设**：ICL 净 +3.5pp
  （救12/坏5），留存修复至 0.88，拯救率 3.8%→7.6%（该轮 retry 拯救 0.000，纯证据效应），
  逐任务 ICL 翻转 升22/降6（p≈0.008）。obs_surprise 0.112→0.197。
- **未解**：内化端未跟上天花板——belief-only 下 uniform 拯救 5.1% vs ours(relu δ) 1.3%
  （8 vs 2，n 小）；嫌疑：2 epoch 训练量不足（loss 降 0.5 行为不动）/ relu 加权覆盖
  173 token vs uniform 全观测面。修法旋钮：步数、on-policy 蒸馏、底座。
- **池子普查**（4 基线样本/任务）：all-fail 154（77%）/ mixed 10 / all-succ 36；
  all-fail 中 14 个被证据臂救活过（证据可穿透死区边缘）。信号区 ≈16–30 任务。
- 协议变更（用户指示）：单种子；动作通道以 P1.6 状态门控进入（v2 起）。

### P1.7 写入强度量效曲线（2026-08-23 终审，tail-200，steps 块+门控动作+近迁移探针）

| 剂量 | ours 拯救 | 恢复率 vs ICL | ours 留存(retry 基线) | ours 净效应 | probe_ours |
|---|---|---|---|---|---|
| 弱 5e-5/α16 (n=117) | 2.9% | 51% | 0.92 (0.83) | **+2** | +1/−1 平 |
| **中 1.5e-4/α32 (n=200)** | 5.7% | 65% | 0.76 (0.93) | −1 | **+8/−6 首现正号** |
| 强 5e-4/α64 (n=200) | 8.2% | **117%（超天花板）** | 0.40 (0.88) | −12 | +14/−17 |

结论：①拯救恢复率在所有剂量 ≥50%（go/no-go 拯救维度达标），强剂量下内化超过
in-context；②留存损伤随剂量增长快于拯救——**无门控时任何剂量都拿不到显著正净效应**
→ 门控不是增强件而是使能件（P3 的存在证明）；③跨实例迁移在中剂量首次转正
（写入足以携带机制、不足以携带破坏）；④定价在所有剂量×指标上支配 uniform
（强剂量下 uniform probe −31 崩盘 vs ours −17）。**P3 trainer 默认 = 中剂量**，
残余遗忘由 G2/G3 + EMA 合并 + NCA 锚定吸收。
（v3 因用户指示提前切换，n=117 为部分样本；v4/v5 全量 200。）

### P1.5 增量 hindsight 仪器化（证据结算曲线）

v0 信号是净位移：Δ(i) = logp(tok_i|结局块) − logp(tok_i|∅)，把三角矩阵 Δ(i,j)
（第 j 步反馈对第 i 步 token 的再定价，j≥i；step1 有 K 个增量、step2 有 K−1 个…）
折叠成一列。telescoping：Σ_j Δ(i,j) = 以全部未来证据为条件的净位移——**训练用净位移
不丢总量**（中途增量含会被后续证据推翻的解释，直接训练 = 自训漂移入口）；丢掉的是
**路径分解**，其正确用途是门控：结算曲线形态（单调早结算 = 稳健事实 vs 振荡晚结算 =
解释层争议）是 token 级信任特征——信念空间的 TD 分解，类比 prioritized replay 用
|TD| 做优先级而训练用 return。振荡路径 = 解释层残差的运行时指纹（衔接 robagent 分界）。

约束（copy-safe 铁律的推广）：增量条件块必须是结算式压缩摘要（outcome-like），
不得含逐字未来观测，否则 obs token 的 delta 退化为复制检测。

落地：worker 已 dump 首答轨迹 + 逐 span deltas（`traj_shard*.jsonl`，s1 起生效）。
s0–s2 完成后抽 20–30 任务离线重打分构造完整 Δ(i,j)，测：① telescoping 校验；
② error/普通 obs 的结算曲线形态差异；③ **路径方差是否预测 ours 臂成败**
（若预测 → 升级为 G1 门控特征）。另一便宜升级待测：块从仅 outcome 扩为全程
结算摘要（仍 2 次 prefill，净信号更富，同样受 copy-safe 约束）。

### P1.6 动作通道定价（2026-08-23 实测裁决）

问题：steps 块把动作原文放进条件后，动作 delta 是复制力（induction 抬升）还是
评价力（坏动作信念下调）主导？实测（v0 vs v1 各 200/121 任务 + v1 轨迹 dump
1352 动作 span）：**评价力胜**——act_gain v0 −0.021 → v1 −0.035（复制未主导）；
按步骤状态分组：ok 步 −0.013 vs ERROR 步 −0.079，评价分辨力 +0.066（复制对两组
共模，组间差分离纯评价）。观测 span +0.061（信念通道健康参照）。

残留问题：32.7% ERROR 步动作 delta 仍为正 → relu 加权会强化 1/3 已知坏动作。
结论：动作通道信号存在（用户直觉正确），但 relu 定价对它太粗。设计（待第 6 臂验证，
v1 三种子之后）：①状态门控 w_act = relu(δ)·1[step ok]；②或中心化
relu(δ − mean δ_act)；③ERROR 步动作可加 unlikelihood 项。v1 系列期间动作通道
保持掩码（系列一致性 + 防 relu 误强化）。

## P2 无门控流式累积（负对照，与 P1 并行）

100–300 任务流，逐任务更新直接合入 session adapter，不门控。
预期先涨后崩。关键产出：**失败曲线 + 毒源归因**——按 mechanism/interpretation
给每次写入打标签，事后归因哪类残差造成干扰（衔接 robagent 结论）。

## P3 门控流式固化（主实验）

协议钉死为 **predict-then-update**：任务 t 的成绩永远由更新前的权重打出，
更新只惠及 t 之后；同一任务不重做（重试仅作为门控触发的 deliberate practice，
其成绩不计入首答曲线）。窗口化执行（W=16–32 任务并发），陈旧度恒为 1 窗口。

五臂对比（同一条流）：不适应 / context 侧累积（全历史 ICL，CL-Bench 证明的最强朴素基线）/
逐任务重置不累积（TT-SI 式）/ 无门控累积（P2）/ 门控累积（ours）。

流的构造分三档：同环境不同实例（迁移上限）/ 同族不同环境（机制层共享，主战场）/
跨域混合（干扰压力测试）。

指标四件套：W-AUC；gain = 有状态 − 无状态；每 4 窗口冻结快照测 held-out ID/OOD；
回访早期任务 + 通用回归套件（GSM8K/MMLU 小子集）。主图 3 种子 × 3 流序。

### P3 冒烟（2026-08-23，首次真卡全闭环）

48 任务流（tail-200 头部）、W=8、中剂量（P1.7 膝点）、G1 台账+G2 重放+G3 探针(4)+EMA(0.5)。
结果：**W-AUC 0.700 vs 冻结基座 0.538**（同任务同序对照，gain +0.16）；w1–w5 每窗高于
基线（均值 0.75 vs 0.55，且 w0 采样劣势 −0.25 起步后全程反超）；逐任务配对 10:4。
**门控击落记录**：5 候选 3 放行 2 拦截——被拦的探针 delta −0.50/−0.25，正是 P1.7
v4 型毒写入；无门控则后段曲线不存在。9 次门控重试、0 hindsight 错误（块预算生效）。
待补：≥3 种子 × 3 流序、五臂全对照（context 累积/无门控/逐任务重置）、扩探针集、
冻结快照 ID/OOD、回访早期任务。

### P3 正式六臂（2026-08-25 凌晨，tail-160 流 W=16 单种子，results/p3m_*）

frozen 31/160、icl（首答带 top-4 检索块）34、**ng（定价+EMA 无门控）20**：前 5 窗 12 vs 8
领先，w4 毒合并后探针 0.75→0.17，进入"循环查询永不结算"（w5–6 32 题全撞 30 步 reward 0）
——**先涨后崩实锤，48 题冒烟 0.700 是崩前截断**。gated（G3 容差 −0.09）拦 6/9、守 v0003，
尾段 6/16（frozen 9）：**门控挡崩不生增益**，12 探针复测噪声 ±0.3 = 信号量级。
paper 剂量（r8 α16 lr5e-4）4 次合并即崩。

**通道归因（`scripts/channel_forensics.py`，两次独立毒合并复现，三变体从父版本训 +12 探针）**：
obs-only 0.167/0.167、act-only 0.667/0.583、both 0.250/0.083（父 0.417）→ **毒源=观测通道 CE**，
权重质量落在病人档案/log_id 等实例数据。推翻"失败轨迹训世界模型最安全"：策略与世界模型
共用 4B LM 时观测 CE 泄漏进策略。→ 主线改 `train_channels=act`。

### P3 800 题流与主方法定型（2026-08-25，results/p3m800_*，语料尾 806 = 800 流 + 6 探针）

臂：frozen / icl_refl / refl_act / refl_signed（全动作 token 带符号死区，不看步骤状态）。
**无 floor 的两条训练臂都崩**：refl_act w6–8 撞 30 步 9/14/15（与 ng 同型，且它无观测通道
→ 观测不是唯一根因）；refl_signed 前 9 窗 17 vs frozen 9 领先，w7/w8 两次"动作全变+探针 0"
合并后退化为只叙述不调用工具（1 步结束）。

三次归因，三个根因，三个修复（均已进代码）：
1. **观测通道背实例数据** → `train_channels=act`。
2. **剂量与证据脱钩**：加权平均 Σw·CE/Σw 让动作质量仅 1.7/0.6 的样本驱动满剂量更新
   （act-only loss 4 步 4→0.001）→ `w_norm_floor`（loss=Σw·CE/max(Σw,floor)，正负各自适用）。
3. **负信用落在框架 token**（`<tool_call>`/JSON 键/工具名）压掉"调用工具"本身 →
   `sediment/semantic.py` 语义掩码：框架 token 任何符号都不给信用，负信用只留值/内容/工具名。

**当前主方法 = 反思增强证据块 + 动作通道带符号残差定价 + 剂量下限 + 语义掩码 + EMA 合并；
门控只量测不执行（保险层）**。具体配置见 README "Main method" 表。对照臂
`p3m800_refl_act_f`（floor + P1.6 状态门）、`p3m800_refl_signed_f`（floor + 语义掩码，
pos_thr 0.05 / neg_thr 0.6）跑中；判据 = w6–11 撞 30 步计数是否抬头 + 成功数 vs frozen。

对分级/连续 reward 的兼容性（为 CL-Bench 准备）：定价本身不依赖成败二值（signed 死区看 δ，
不看 reward）；块里已渲染数值 `reward=`；唯一残留二值点 = `SUCCESS_THRESHOLD=0.999` 决定的
`SUCCEEDED/FAILED` 标签与重试触发。迁移到连续 reward 只需把标签改为相对位置
（如"高于/低于 buffer 同域历史均值"），并复核 δ–reward 相关是否仍在（P0 为 +0.336）。

## P4 外部对标（只挑三个）

1. **ALFWorld**：aTTT 在此 +5.0，直接可比；底座选成功率落在 mixed 区间的
   （aTTT 的 4B 在 ALFWorld 仅 1.9%，无物可学，或需 9B 级）。
2. **LifelongAgentBench**：唯一技能依赖任务流终身基准；LifeSkill +7 是要打的数。
3. **tau2-bench**：截至 2026-08 零篇适应类论文使用，占空位 + OOD 章节。

4. **CL-Bench（2606.05661，2026-08-25 评估：适合，排在 aTTT 对标后、tau2 前）**：
   协议与我们一字不差（固定顺序流 20–120 episode/域、gain = 同实例 stateful − stateless、
   归一化 gain 除 headroom）；潜在结构是机制层的（schema 约定/代码库布局/对手策略）且
   5/6 域有 migration 测漂移；反馈程序化可验证（bash 报错/SQL 结果）。**它明说不评参数方法、
   征集社区提交**，上下文侧全体（ICL/Notepad/Mem0/ACE/Claude Code/Codex）被钉在归一化 gain
   25.4% 天花板，结论"agents overfit to immediate observations"与我们观测通道归因互证。
   约束：只评前沿模型（无开源小模型数字），4B stateless 可能贴 0 → 先 frozen 摸底挑有 headroom
   的域（Database Exploration、Cohort Studies 最像 EnvScaler 工具调用；Poker/Sales 可能全废）；
   reward 连续（见 P3 兼容性说明）；Codebase 域要 Docker，每域一个 env adapter。
   计划：先 2 域三臂 frozen / ICL 全历史（它自己的 SOTA 基线）/ ours。

SWE-bench Lite 为 stretch；GAIA 不做（不可重复交互、无训练集）。

## 横切纪律

- compute-matched：ICL 臂给等额算力（同预算 best-of-N）；每方法报 rollout 数 +
  梯度 FLOPs + 相对开销倍数（aTTT 报 1.9×，我们也报）。
- 流式方法对任务顺序敏感：单一顺序的曲线不可信。
- Judge/门控与 actor 不同模型。
- 每次写入落 jsonl 台账（task ids、w_t 统计、门控决策、adapter 版本谱系），
  供 P2/P3 的归因分析与论文 forensics。
