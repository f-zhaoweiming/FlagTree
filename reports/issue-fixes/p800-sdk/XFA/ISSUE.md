# SDK01 · kunlun_ops 与 XPyTorch 的 XFA 二进制接口不匹配，导致整包导入失败

在 P800 上，Python 3.10 和 Torch 2.9.0+cu129 可以正常启动。加载公开的 `kunlun_ops 0.1.58+ee39020a` 时，动态加载器在 `libxops_blocks.so` 中找不到 `dsa_mqa_logits` 的旧签名。导入在任何算子启动前中断，连 ReLU 等不使用该函数的 Python 入口也无法访问。

触发环境中的 XPyTorch 构建为 `bc1b1dc6f`。调用方传入的最后两项是布尔值，运行库的对应导出还要求两个指针。头文件为指针提供了默认值，已编译的调用方仍依赖旧符号。只检查 `libxops_blocks.so` 时，94 个 XFA 引用缺 29 个。检查旧 wheel 内全部 ELF 并去重后，106 个引用缺 33 个。[原始导入与符号回执](verify-original.json)

**复现**　在安装了 XPyTorch 的 Python 3.10 环境中，将本目录的两个脚本放在一起。`$SCRATCH` 指向已有的个人临时目录。脚本将 wheel 解压到临时目录并启动新的 Python 进程，输出动态加载器的报错、实际提供符号的库和未解析清单。预期结果是导入成功并返回 0，故障组合返回 1。

```bash
python reproduce.py /path/to/kunlun_ops-0.1.58+ee39020a-cp310-cp310-linux_x86_64.whl "$SCRATCH"
```

旧版 wheel 来自厂商面向 Torch 2.5.1 的安装说明。Torch 2.9 的安装说明给出了另一组成对发布的 SDK 和 wheel。[2.5.1 安装说明](https://github.com/baidu/vLLM-Kunlun/blob/main/docs/source/installation.md) · [2.9 安装说明](https://github.com/baidu/vLLM-Kunlun/blob/v0.25.1-dev/docs/source/installation.md) · [复现脚本](reproduce.py)
