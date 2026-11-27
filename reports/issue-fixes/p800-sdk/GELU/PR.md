# SDK03 · 在 FP16 GELU 的公开写出入口使用 XPyTorch exact 实现

关联本地 [Issue SDK03](ISSUE.md)。在设备、dtype 和形状匹配的 FP16 输入与输出缓冲区上，`kunlun_ops.gelu` 调用 `torch.ops.aten.gelu.out(..., approximate="none", out=y)`，直接写入调用方提供的张量并返回状态 0。发布包内的原始机器码用于修复前对照，个人环境加载经过校验的 Python 包装层。[代码补丁](fix.patch)

**验证**　四个正式形状的完整输出均通过原始 FP32 参考下的三项精度指标。枚举全部 65,536 种 FP16 位型后，非 NaN 输出与 FP16 eager 逐位一致，NaN 位置完全一致，非连续原地输出也通过。此次修复使用 XPyTorch 的 ATen 路径，耗时记录覆盖整个 Python 入口。[性能和精度明细](../FP16/p800-results.json) · [边界回归](../FP16/contract.json)

此补丁以官方 wheel 解压目录为根。可单独应用，在该环境中运行 `python verify_wrappers.py --operator gelu`。个人 Docker 的准备程序在提取固定 SHA-256 的 wheel 后应用同样的源码变更，并记录实际模块指纹。[准备程序](../XFA/prepare_ops.py)

两轮独立进程的端到端加速比范围为 **0.933–0.996×**，计算口径为同卡 FP16 eager 同步耗时除以修复后公开入口同步耗时。[两轮逐项原始指标](../FP16/p800-results.json)
