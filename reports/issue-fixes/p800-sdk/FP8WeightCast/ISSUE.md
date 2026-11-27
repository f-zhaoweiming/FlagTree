# SDK06 · P800 原生 eager 无法将 float8_e4m3fn 权重转换到 bfloat16

**归属**　昆仑芯 XPyTorch / XDNN 的 FP8 cast 分派。原始模型运行关闭 FlagGems 和图优化。

## 观测与期望

权重加载器对 [2048,2048] 的 float8_e4m3fn 张量执行 to(bfloat16)。XDNN 将源类型打印为 UnknownScalarType，在 cast.cpp:163 和 XMLIR copy_kernel.cpp:407 返回 4。反量化乘 weight_scale 在此之后。

映射正确的 FP8 标量类型并实现格式转换。覆盖有限值、零和特殊编码，并在保留发布者 weight_scale 的情况下校验完整权重加载与前向。

## 已有处理方案与范围

D02 处理 FlagGems 的非连续 copy_。M96 针对 NVFP4 打包权重与模型加载器。两项所处理的输入格式和调用路径均不同。

## 受影响模型与证据

| 模型 | 固定 revision | 原始日志 | 定位实验 |
|---|---|---|---|
| nvidia/llama-nemotron-embed-vl-1b-v2-fp8 | b4751b329b7d797cb6abfe018e8bedf1051a80aa | [eager 日志](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.179.154.78/92270bad15fc281e26c5/main/local-off/console.log) | [同版本重放](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.179.154.78/92270bad15fc281e26c5/main/eager-dtype-replay.json) |

## 修复验收

复现的镜像、包、设备和图配置见上述原始运行状态。调用方规避与实卡端到端复测进展见[本次逐模型修复表](../../P800-eager-修复对照-20261126.md)。厂商原生实现的修复仍需在无调用方规避的条件下重跑。
