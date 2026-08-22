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
