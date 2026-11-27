# SDK01 · 为 kunlun_ops 建立与运行库配套的独立加载配置

关联本地 Issue [SDK01](ISSUE.md)。修复位置是 P800 接口测试的依赖准备和启动入口。

依据厂商 Torch 2.9 安装说明，使用同批的 `kunlun_ops 0.1.227+2b100f96` 和 XPyTorch `953ed40fc`。安装器校验两个发布文件的 SHA-256，在个人目录解包 SDK、准备独立 Python 环境。在新进程中实际导入 Python 包和两个扩展，并从进程已映射的 XFA 库逐一解析整个 wheel 的强未定义 XFA 符号。校验通过后原子更新运行配置。测试入口使用同一解释器，并再次核对实际加载库的指纹。[构建代码](build_profile.py) · [准入代码](prepare_ops.py) · [完整代码差异](fix.patch)

| P800 上的组合 | 所需 XFA 符号 | 缺失 | 实际导入 |
|---|---:|---:|---|
| 旧 wheel + 镜像原 SDK | 106 | 33 | 失败 |
| 新 wheel + 镜像原 SDK | 176 | 116 | 失败 |
| 新 wheel + 配套 SDK | 176 | 0 | 成功 |

相同 Torch 版本下，只更换 wheel 仍然会失败。负对照也验证了失败配置不会被发布。[旧组合](verify-original.json) · [只换 wheel](verify-mixed.json) · [配套组合](verify-paired.json)

**设备验证**　在同一张空闲 P800 上通过正常入口测试 7 个 Python API、32 个组合。24 项产生完整输出和配对耗时。其中 ReLU、SiLU、matmul、matmul_v2、bmm 共 20 项通过已有的相对 FP16 eager 精度门槛。GELU 4 项取得结果，在该门槛下未通过。其余 8 项在 add_layernorm 自身报出 `add_layernorm not support x/weight/output dtype`，触发条件为本次 FP16 输入、权重和输出。该限制与导入故障分开登记。设备统计采用运行期间未发现其他进程的一轮。[P800 逐项结果](p800-results.json)

**重跑**　先从厂商 2.9 安装说明下载两个发布文件。在个人 Docker 中将下述变量设为各自路径。`build_profile.py` 最后一行给出独立解释器的路径，填写为 `$PROFILE_PYTHON` 后执行第二条。之后可以通过原有 `ops` 基准入口复测。

```bash
python build_profile.py "$XPYTORCH_INSTALLER" "$PERSONAL_DIR"
python prepare_ops.py --wheel "$KUNLUN_WHEEL" --python "$PROFILE_PYTHON"
```

**后续复核**　GELU 和 Add+LayerNorm 的包装层修复已在同一配套环境完成，两轮单卡各 32/32 项通过。原版的逐项结果保留用于重现上述两个数值或 dtype 问题。[GELU 修复](../GELU/PR.md) · [Add+LayerNorm 修复](../AddLayerNorm/PR.md) · [修复后设备结果](../FP16/p800-results.json)
