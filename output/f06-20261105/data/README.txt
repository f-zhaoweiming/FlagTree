F06 阶段报告数据包，2026-11-05

所有时间单位为微秒，speedup=基线耗时/候选耗时。
P800 216 自动路径含169回退，不能全部计为融合成功。
RTX 4090 213/216 F06通过，Inductor155/216通过。
单乘并行专项36例中28例计时、4例完整验收通过。all_gates=false的时间只用于诊断。
P800特殊值实验的status及special_passed必须同时查看，不能仅按有时间字段判定正确。
不同批次不混合求加速比；后续扩展检查不由历史passed替代。
CSV使用UTF-8 BOM，JSON保留原始采样与详细原因。
source_manifest.json列出输入证据的SHA256。
