# SDK02 · 单卡 XFFT 为复数 FP16 创建一维计划时返回 14

KL3 XFFT 20251230 在 P800 上处理连续、单批、秩为 1 的复数输入时，长度 128、512、1024 的 `cufftCreate` 均成功。将输入、输出和执行类型都设为复数 FP16 后，`cufftXtMakePlanMany` 对三个长度均返回 `14 / CUFFT_NOT_IMPLEMENTED`，计算尚未发射。

在这版 x86-64 主机库中，三个类型编号均为 6 时，`0x1f68a` 附近的分派把控制流导向 `0x1f6b3` 返回 14。复数 FP32 的类型组合 4/4/4 可进入 C2C 计划器。这个边界适用于本次库版本和布局。[发布包](https://klx-sdk-release-public.su.bcebos.com/xpufft/kl3/20251230/xpufft_ubuntu2004-x86_64.tar.gz) · [原始计划返回码和设备结果](p800-results.json)

**复现**　在安装 XPyTorch 的单卡 P800 环境中，把 `verify_xfft.py` 和 `xfft_fp16.py` 放在同一目录。传入发行包中实际的 XFFT 动态库。脚本对每个长度先建原生 FP16 计划，保留 `native_status`，并给出后续实际执行类型和正逆变换结果。

```bash
python verify_xfft.py "$XFFT_LIBRARY"
```
