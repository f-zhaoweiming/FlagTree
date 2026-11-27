# SDK05 · P800 原生 eager 对 complex64 除实数标量时误入未支持的 cast

**归属**　昆仑芯 XPyTorch / XDNN 的复数标量运算与类型转换。原始模型运行关闭 FlagGems 和图优化。

## 观测与期望

模型实际执行 rfft(n=108)，得到 XPU complex64 [768,55]。该张量除以 Python 整数 108，XDNN cast.cpp:163 选择 complex64 → float32，并在 XMLIR copy_kernel.cpp:407 返回 4。

支持复数张量按实数缩放并保留实部、虚部及 complex64 结果。分别校验除以 108、模型的后续频域乘法和逆变换，再核对三种模型的端到端输出。

## 已有处理方案与范围

M06 记录了 32k 的原始组合表达式和 CPU 参考。SDK02 的绕行针对 FP16 复数 FFT 计划返回 14，本例已成功取得 FP32 输入的 RFFT 结果。两项均未修复本例标量除法。

## 受影响模型与证据

| 模型 | 固定 revision | 原始日志 | 定位实验 |
|---|---|---|---|
| togethercomputer/m2-bert-80M-2k-retrieval | 0d5e23e332fc54a2dc445636fd4f0821c78fbeb1 | [eager 日志](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.138.5.189/284683104f8008f8cc6e/main/local-off/console.log) | [同版本重放](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.138.5.189/284683104f8008f8cc6e/main/eager-dtype-replay.json) |
| togethercomputer/m2-bert-80M-8k-retrieval | 90e0b28f3382c289cc6c1e92ef53d7dc5a3ec14b | [eager 日志](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/81.70.153.29/509cf4906e59c785b527/main/local-off/console.log) | [同版本重放](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/81.70.153.29/509cf4906e59c785b527/main/eager-dtype-replay.json) |
| togethercomputer/m2-bert-80M-32k-retrieval | a2ccdc5b5661a282c77545e586a019f387ab7a48 | [eager 日志](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.179.154.78/c264566c5b0f2208f847/main/local-off/console.log) | [同版本重放](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.179.154.78/c264566c5b0f2208f847/main/eager-dtype-replay.json) |

## 修复验收

在个人容器中，用固定 revision 的发布者 `fftconv_ref` 对实数傅里叶卷积实现做了 16 组 CPU 数值对照，覆盖 FP32、FP64、4 个输入长度、反向卷积核、GELU 和掩码。最大绝对误差为 1.10×10⁻⁵。三种上下文版本均已用正式输入在 P800 完成端到端规避复测。正式输入的短序列未覆盖 2k、8k、32k 的满长计算。[数值对照记录](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.179.154.78/eager-dft-cpu-equivalence.json)

复现的镜像、包、设备和图配置见上述原始运行状态。调用方规避与实卡端到端复测进展见[本次逐模型修复表](../../P800-eager-修复对照-20261126.md)。厂商原生实现的修复仍需在无调用方规避的条件下重跑。
