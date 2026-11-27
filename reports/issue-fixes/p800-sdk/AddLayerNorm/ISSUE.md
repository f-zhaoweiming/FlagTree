# SDK04 · Add+LayerNorm 拒绝文档声明支持的全 FP16 组合

在 P800 配套 XPyTorch 2.9 和 `kunlun_ops 0.1.227+2b100f96` 上，按该 wheel 的 Python 文档向 `add_layernorm` 传入 FP16 的两个输入、权重、偏置、主输出和残差输出，四个形状的随机及高偏置用例全部报 `add_layernorm not support x/weight/output dtype`。

在 `xpu_flash_ops` 的主机二进制中，FP16 分支在 `0x2895d3` 检查权重的类型码 6（FP32），权重是 FP16 时在 `0x2895db` 跳到报错路径。权重和偏置从 FP16 无损提升为 FP32 后，同一个入口可运行，八项的残差输出均通过，但归一化输出仍有至少一项精度指标劣于 FP16 eager。最小的随机用例有 142 个归一化元素与 FP16 eager 不同。[机器码与独立对照](../FP16/diagnosis.json)

**复现**　在包含原始 wheel 的单卡环境运行下列命令。脚本通过公开入口调用 Add+LayerNorm，并以相同的 FP16 输入构造厂商 FP32 融合对照。

```bash
python verify_wrappers.py --operator add_layernorm
```

[复现脚本](../FP16/verify_wrappers.py)
