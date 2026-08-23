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

## P4 外部对标（只挑三个）

1. **ALFWorld**：aTTT 在此 +5.0，直接可比；底座选成功率落在 mixed 区间的
   （aTTT 的 4B 在 ALFWorld 仅 1.9%，无物可学，或需 9B 级）。
2. **LifelongAgentBench**：唯一技能依赖任务流终身基准；LifeSkill +7 是要打的数。
3. **tau2-bench**：截至 2026-08 零篇适应类论文使用，占空位 + OOD 章节。

SWE-bench Lite 为 stretch；GAIA 不做（不可重复交互、无训练集）。

## 横切纪律

- compute-matched：ICL 臂给等额算力（同预算 best-of-N）；每方法报 rollout 数 +
  梯度 FLOPs + 相对开销倍数（aTTT 报 1.9×，我们也报）。
- 流式方法对任务顺序敏感：单一顺序的曲线不可信。
- Judge/门控与 actor 不同模型。
- 每次写入落 jsonl 台账（task ids、w_t 统计、门控决策、adapter 版本谱系），
  供 P2/P3 的归因分析与论文 forensics。
