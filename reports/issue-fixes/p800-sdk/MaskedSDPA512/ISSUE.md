# SDK07 · P800 原生高效 SDPA 拒绝带布尔掩码的 512 维注意力

**归属**　昆仑芯 XPyTorch / XMLIR 原生高效注意力。原始模型运行关闭 FlagGems 和图优化。

## 观测与期望

第六次 SDPA 调用的 Q/K/V 为 FP16 [10,4,40,512]，布尔掩码为 [10,1,40,40]，scale=1，causal=False。原生高效路径在 _efficient_attention_forward.cpp:340 返回 INVALID PARAMETER。

高效实现应支持该组合，或由原生分派为不支持的形状保留原掩码、scale 并回退 math。用同一输入校验全量输出，再复测整模型。

## 已有处理方案与范围

D06、D08、D09 修补 FlagGems ON 的 bias、掩码构造及非方形边界。M51 记录 Meta 与真实输出步幅差异。本例关闭 FlagGems 和图优化。同一输入强制 math 已在算子层通过，无须移除掩码。

## 受影响模型与证据

| 模型 | 固定 revision | 原始日志 | 定位实验 |
|---|---|---|---|
| google/embeddinggemma-2 | f9972ac22e696a105c482202646979e33e4b5a1e | [eager 日志](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/81.70.153.29/39b44716f6fa2e5010de/main/local-off/console.log) | [同版本重放](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/81.70.153.29/39b44716f6fa2e5010de/main/eager-intermediate-replay.json) |

## 修复验收

复现的镜像、包、设备和图配置见上述原始运行状态。调用方规避与实卡端到端复测进展见[本次逐模型修复表](../../P800-eager-修复对照-20261126.md)。厂商原生实现的修复仍需在无调用方规避的条件下重跑。
