# 选择语言 / Choose a language

## 离线示例 / Offline demo

Python 3.11+，建议在独立虚拟环境内执行：

```sh
python -m pip install -e .
python -m evidence_agent_rl.demo
python -m unittest discover -s tests -v
```

示例使用虚构资料和固定动作，在 CPU 上运行，不调用模型服务。
它演示证据引用校验、违规引用拒绝与轨迹重放，不代表真实模型表现、
语义蕴含验证或已完成的 RL 训练。完整研究流程及外部数据未在此示例验证。
输出为 JSON；需要保存时可显式使用 `--output demo-report.json`，已有文件不会覆盖。
贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

- [简体中文](README.zh-CN.md)
- [English](README.en.md)


---
This code-only snapshot excludes model weights, datasets, logs, and credentials. Licensed under MIT; see [LICENSE](LICENSE).



