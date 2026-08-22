# 文献坐标（截至 2026-08，三路调研压缩版）

## 1. 参数化 TTT 配方全景

主线规律：**LoRA/窄参数 + 每条数据 1–32 步 + 逐实例重置**。重置是因为累积会崩。

| 谱系 | 代表 | 配方要点 |
|---|---|---|
| Few-shot 逐任务 | ARC TTT (2411.07279) | LoRA r=128, LOO 伪任务+增广 ≤250 例, AdamW 1e-4, 2ep, 逐任务重置 |
| 测试时 RL | TTRL (2504.16084) | 64 rollout 多数投票伪 reward, GRPO lr 5e-7, 整测试集累积 |
| 置信度系 | RLSC (2506.06395), EM (2505.15134), Intuitor (2505.19590) | 16 样本 10–20 步; 熵/自信度做 reward |
| 检索式 | TTT-NN (2305.18466), SIFT (2410.08020) | 20–50 邻居各 1 步, 由近及远; SIFT 不确定性选点+自适应停机 |
| 长上下文内化 | LIFT (2502.14644), qTTT (2512.13898), PERK (2507.06415) | qTTT 仅更 W_Q 32 步, FLOP 对齐胜 thinking tokens; PERK 元学习内环 4 步 |
| 上下文蒸馏 | Cartridges (2506.06266), LoRA 记忆库 (2605.28889), NCA (2606.11627) | KL(带上下文 teacher ‖ 裸 student); 模块化 adapter 库+检索挂载+熵门控; NCA 修上下文回归退化 |
| Agent 参数 TTT | aTTT (2607.03441), TT-SI (2510.07841), TMEM (2606.04536), LifeSkill (2606.04815) | 见下节 |
| 架构线（避开，借件） | TTT-layers (2407.04620), Titans (2501.00663), LaCT (2505.23884), ATLAS (2505.23735), TTT-E2E (2512.23675) | 惊讶+动量+遗忘门; Muon 内环; 只更最后 1/4 MLP 层保稳定; per-token 可学习 lr |

可直接抄的组件：TTT-E2E 的窄参数子集；per-token lr（w_t 即其 agent 版）；
Titans 门控三件套（只有幅度门无价值门——差异位）；NCA 锚定 KL；
2605.28889 读侧熵门控；SIFT 停机准则。

## 2. aTTT 机制细节（最近邻竞品）

每 K=5 环境步更新一次；候选文本三选一：Self（最近一步推理+动作）/ Env（最近观察）/
Summary（LLM 压缩进度笔记）；**只用最近一步不用全前缀**。纯 NTP 无 reward；
n-gram 降权 w=max(0.05, 1/(1+f))，f=3-gram 在先前更新史的出现次数。
LoRA r=8 α=16 lr 5e-4 每次 2 步；异步训练 GPU + vLLM 热换 <100ms 不阻塞；
**episode 内持续、episode 间重置**。ALFWorld 9B 50.7→55.7；SWE-Lite 27B 57.8→62.7；
增益集中于"有底力但长轨迹漂移"。定性：episode 内防漂移自稳定器，非学习器
（零残差自模仿；唯一门控是反向的重复抑制）。

## 3. 崩溃证据链（C3 的文献支撑）

- 2605.28889：累积蒸馏互相覆盖，SQuAD EM 0.00
- SEAL (2506.10943)：连续 self-edit 灾难性遗忘（自报）
- SRT (2505.21444)：纯自奖励持续训练必然 reward hacking 坍缩
- TTRL-Guard (2605.19444)：正确答案灭绝窗口，多数投票不可逆压制少数正确
- SEAGym (2606.17546)：频繁更新不提升 held-out；快照后期崩坏
- Beyond Perplexity (2607.00368)：loss 降但自由回忆为零——行为学验证的依据
- 批判线：2603.12875（TTRL 对齐揭示基准熟悉度伪影）

## 4. Agent 侧适应版图（2025 上下文侧 → 2026 参数侧分水岭）

上下文/记忆侧（全部冻结权重）：ACE (2510.04618, AppWorld 63.7→76.2)、
ReasoningBank (2509.25140)、Memento (2508.16153, GAIA 79.4%)、Agent KB (2507.06229)、
AWM (2409.07429)、Dynamic Cheatsheet (2504.07952)、Training-free GRPO (2510.08191)。
反方证据：CL-Bench (2606.05661) 朴素长上下文 ICL 打赢 ACE/Mem0；
EvoMemBench (2605.18421) 长上下文很能打。

训练时无奖励内化（近邻）：Early Experience (2510.08558, Meta)——implicit world
modeling = 训练时版 belief 残差；Learn-by-Interact (2501.10893)。

参数侧 2026：aTTT / TT-SI / TMEM / LifeSkill / Meta-TTL (2604.00830, W-AUC 指标) /
OLIVIA (2605.11169, bandit 头) / Salesforce TTA (2511.04847, 免梯度)。
综述：Adaptation of Agentic AI (2512.16301)。

## 5. 基准 × 适应方法矩阵（选型依据）

| 基准 | 已被谁用 | 备注 |
|---|---|---|
| ALFWorld | Reflexion/ExpeL/Early Experience/**aTTT** | 最便宜的对标场 |
| SWE-bench Lite/Verified | **aTTT**/Agent KB/ReasoningBank/DGM | 贵，stretch |
| LifelongAgentBench (2505.11942) | LifeSkill (+7) | 唯一技能依赖任务流终身基准 |
| CL-Bench (2606.05661) | ACE 被评 8.6% gain | gain = 有状态−无状态 |
| AppWorld | ACE（offline+online 双协议） | 有 train/test-normal/challenge |
| tau2-bench | **零篇适应类论文** | 空位 |
| GAIA | Memento/Agent KB/EvoMem | 不可重复交互，不适合参数 TTT |
| StreamBench (2406.08747) | 单轮任务流 | 协议参考 |
| SEAGym (2606.17546) | 评 ACE/TF-GRPO | 冻结-更新-验证协议参考 |

## 6. 本项目引用位姿

- Intro 三段：上下文侧饱和（CL-Bench+SEAGym+robagent）→ 参数侧兴起但全员重置
  （aTTT/TT-SI/配方全景）→ 安全累积缺门控（崩溃证据链）。
- 区分度句：TTRL 组内统计不可部署 vs 我们 G=1；aTTT reset 处是我们的起点。
- 理论邻居：Nested Learning/CMS 多频更新（NeurIPS 2025）对应 Tier 梯子。
