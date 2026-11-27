# SDK03 · FP16 GELU 的旧 XDNN 入口与模型要求的 exact GELU 存在数值差异

在 P800 配套的 XPyTorch 2.9 和 `kunlun_ops 0.1.227+2b100f96` 上，`kunlun_ops.gelu(x, out)` 可以执行。对四个固定形状，由同一组原始 FP32 输入生成 FP16 输入后，四项相对原始 FP32 eager 的 RMSE 均高于 FP16 exact eager。最小形状 16×64 的 1,024 个输出中有 142 个与 FP16 exact eager 不同，RMSE 分别为 0.000196129 和 0.000194918。

Python 包调用 `xpu_kunlun_ops.gelu`。其主机二进制在 `0x150494` 调用原有六参数 `api::gelu<float16>`。使用厂商另一个带近似模式参数的重载并显式传入 `false`，四个形状的输出仍与旧入口相同。对全部 63,488 个有限 FP16 输入，旧入口与 FP16 exact eager 有 24,277 个不同值，与 tanh eager 有 27,339 个不同值。这把差异定位到了所调用的厂商实现；这些数据不足以确定闭源设备核内部第一条造成误差的指令。[二进制与同输入对照](../FP16/diagnosis.json)

**复现**　在配套的单卡 XPyTorch 环境中加载原 wheel，运行下列脚本。它枚举 FP16 位型，验证输出缓冲区和非连续原地调用。修复前可观察到非 NaN 输出的差异。

```bash
python verify_wrappers.py --operator gelu
```

[复现脚本](../FP16/verify_wrappers.py) · [PyTorch exact 与 tanh GELU 定义](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GELU.html)
