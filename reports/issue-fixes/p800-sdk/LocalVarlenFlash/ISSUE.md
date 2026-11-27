# SDK08 · P800 BF16 变长 FlashAttention 在局部窗口 (64,64) 触发设备访存越界

**归属**　昆仑芯 XPyTorch 预编译 libxpu_flash_attention.so / XPU3 FlashAttention。原始模型运行关闭 FlagGems 和图优化。

## 观测与期望

真实 QKV 为 BF16 [282,3,12,64]，最长序列为 67。第二次 attention 对同一份 QKV 和 cu_seqlens 使用全局窗口能同步，恢复局部窗口 (64,64) 后返回异步错误 700。原运行的 12 个 SDNN cluster 报告 reason[4]。

修复变长局部窗口的内存边界。校验短于和长于窗口的序列，输出与保留局部掩码的独立计算对齐。复测须在调用后同步。

## 已有处理方案与范围

M11 的特定模型策略可避开部分 FlashAttention。M97 的去填充和 M102 的 FakeTensor 步幅属于图捕获。把局部窗口改为全局窗口会更改模型语义。驱动装载地址和已提取机器码尚不能标出具体越界指令。

## 受影响模型与证据

| 模型 | 固定 revision | 原始日志 | 定位实验 |
|---|---|---|---|
| lightonai/DenseOn | cb9947ebccb33862d24e3c7ca2edb25e51acd887 | [eager 日志](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/81.70.153.29/abde0d4fb64fc29dcd3b/main/local-off/console.log) | [同版本重放](../../../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/81.70.153.29/abde0d4fb64fc29dcd3b/main/eager-flash-window.json) |

## 修复验收

复现的镜像、包、设备和图配置见上述原始运行状态。调用方规避与实卡端到端复测进展见[本次逐模型修复表](../../P800-eager-修复对照-20261126.md)。厂商原生实现的修复仍需在无调用方规避的条件下重跑。
