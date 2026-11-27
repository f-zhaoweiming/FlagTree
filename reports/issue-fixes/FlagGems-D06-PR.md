# [FlagGems][Kunlunxin] 在 P800 高效注意力前向中支持非空 bias

本地 PR 审阅稿，目标为 `flagos-ai/FlagGems` 的 `master`。对应本地 D06。原来的 FlagTree 编译器问题继续使用独立文档。[Issue 草稿和已有方案](FlagGems-D06-Issue.md) · [FlagTree Issue 与 PR](8项Issue与PR审阅全文-20261119.md)

## 修改

将 D06 的两个注意力文件从已验证的本地补丁移植到官方提交 `569edb9713689a7baab50ba5245dae8790ffa335`。为双遍注意力准备可广播的掩码缓冲区，组合 causal 与 bias，保留屏蔽位置的负无穷语义。修正全屏蔽行的归一化和 log-sum-exp。Philox 占位张量显式在 CPU 上创建。新增算子测试覆盖 FP16、FP32、布尔掩码、非连续切片、负偏置、全屏蔽行及 causal 与 bias 组合。[修复分支代码](../../work/2026-11-24/xpu-gems-eager/FlagGems-fix/src/flag_gems/runtime/backend/_kunlunxin/ops/_scaled_dot_product_efficient_attention.py) · [测试代码](../../work/2026-11-24/xpu-gems-eager/FlagGems-fix/tests/test_scaled_dot_product_efficient_attention.py)

## 已有处理和本次验证的区别

历史组合包同时包含跨设备 `copy_` 和 `randperm` 修复。本 PR 单独移植 D06。D08 的一维掩码性能优化、D10 的 LayerNorm 与矩阵路径有独立验证记录。D09 的非方形输入仍按原证据记录。[历史补丁及回归](../../work/2026-11-19/flaggems-official-fixes/README.md) · [D08 性能及边界](../../work/2026-11-19/flaggems-attention-opt/README.md)

11 月 25 日，在原 MiniLM 主测的同一张 P800 卡上，仅挂载历史 D06 的两个注意力文件，模型前向通过，输出相对 eager 的最大绝对误差为 0.000244141。对同一移植做严格算子回归时，本地编译器在 28 项中有 4 项全屏蔽行出现 NaN，官方编译器通过 28 项。进一步定位到分块计算额外分配的 K 尾部位置。原代码在注意力分数缓冲区给这些位置填入 `-1e30`，全屏蔽行的实际行最大值还要更小，故这些位置可能在指数运算产生无穷大。本 PR 把分数缓冲区预填值改为负无穷，使尾部贡献为零。模型输入的 padding 掩码保持原义。[算子与模型对照记录](../../work/2026-11-24/xpu-gems-eager/evidence/d06-review-summary.json)

累计包 `p800fix5` 已把同一项尾列修正合入 D08 的一维掩码实现，并保留 D10。D06 的 28 项回归在这套本地包上通过。整模型验证使用与部署相同的修复包，进程记录实际加载的算子和编译器摘要。[累计源码与 wheel 的逐文件核验](../../work/2026-11-24/xpu-gems-eager/all-local-fixes/source-receipt.json) · [累计版本清单](../../work/2026-11-24/xpu-gems-eager/all-local-fixes/pins.json) · [算子回归](../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.179.154.78/e9e2c8815f70dad48afc/operator-regression/local-on/pytest.xml) · [MiniLM 运行记录](../../work/2026-11-24/xpu-gems-eager/all-local-fixes/evidence/43.179.154.78/e9e2c8815f70dad48afc/main/local-on/state.json)

## 实测与补丁

| P800，关闭图优化 | 本地 FlagTree | 官方 FlagTree |
|---|---|---|
| 本 PR，28 项独立算子回归 | 28/28 通过 | 28/28 通过 |
| 本 PR 的两个注意力文件，MiniLM 完整前向 | 通过，最大绝对误差 0.000244 | 通过，最大绝对误差 0.000305 |
| 本 PR，带掩码 SDPA，FP16，原生 / ON | 0.0310 / 1.0599 ms | 0.0315 / 1.0423 ms |
| 本 PR，带掩码 SDPA，FP32，原生 / ON | 0.0279 / 1.0779 ms | 0.0277 / 1.0202 ms |
| 用户现有 `p800fix4`，MiniLM 完整前向，原生 / ON | 6.589 / 16.209 ms，最大绝对误差 0.000244 | 6.968 / 16.080 ms，最大绝对误差 0.000305 |

算子回归以 CPU double 为参考。MiniLM 独立复测在原 FlagGems 包上仅挂载 PR 的两个注意力文件，两版 ON 进程均核对了算子源码指纹。算子性能用例采用 Q、K、V 形状 `[10,12,54,32]`，掩码为 `[10,1,54,54]`。保留官方 benchmark 的 `kernel` 计时模式，按 `--warmup 3 --iter 10` 运行，两版都通过。最后一行为全清单的正式修复包，按各自同卡原生前向校验，3 次预热、10 次同步前向取均值。[算子回归原始记录](../../work/2026-11-24/xpu-gems-eager/evidence/d06-review-records.tgz) · [两版算子计时及独立模型复测](../../work/2026-11-24/xpu-gems-eager/evidence/d06-review-summary-43.179.154.78.json) · [全模型修复包指纹](../../work/2026-11-24/xpu-gems-eager/fixed-package/pins.json) · [逐模型实测](../../work/2026-11-24/xpu-gems-eager/fixed-package/model-results.csv)

本地分支 `codex/kunlunxin-efficient-attention-bias`，提交 `1e68c2b50`。补丁能应用于干净的官方基线 `569edb971`，四个改动文件均通过 pre-commit。全仓钩子另报已有的重复定义、格式和元数据问题。[可应用补丁](FlagGems-D06.patch) · [新增算子性能用例](../../work/2026-11-24/xpu-gems-eager/FlagGems-fix/benchmark/test_scaled_dot_product_efficient_attention.py) · [改动文件检查](../../work/2026-11-24/xpu-gems-eager/d06-precommit-changed.log) · [全仓检查](../../work/2026-11-24/xpu-gems-eager/d06-precommit.log)
