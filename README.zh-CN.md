# EvidenceAgent-RL

> 代码快照发布说明：仅含筛选源码，不含模型、数据、日志或凭据。

EvidenceAgent-RL 是一个围绕论文证据问答的 Agentic-RL 研究工程，研究如何把证据边界、检索策略记忆和工具交互纳入可审计的训练与评测流程。

## 离线快速开始

使用 Python 3.11+，在独立虚拟环境中执行：

```sh
python -m pip install -e .
python -m evidence_agent_rl.demo
python -m unittest discover -s tests -v
```

此示例只用虚构资料和固定动作，在 CPU 上运行，不需要模型服务或凭据。
它用于验证引用边界与确定性重放，不是模型基准测试、语义蕴含验证或 RL 训练结果。
完整研究流程及外部数据不在此次验证范围内。
JSON 默认输出到终端，`--output demo-report.json` 仅创建新文件，不覆盖已有文件。
见 [贡献说明](CONTRIBUTING.md)。

## 研究工程（已有快照）

- 检索与工具调用环境
- 证据约束的策略/记忆机制
- 训练、冻结选择和评测门禁脚本
- 失败分析与结果审计工具

## 数据与结果

公开仓库应只含源码、通用配置和获准公开的示例。正式测试集、私有语料、轨迹、checkpoint 和证据包均须排除。所有分数按对应实验协议解释，不把诊断结果描述成正式 holdout 成绩。

## 语言

English: [README.en.md](README.en.md)

## 许可

代码采用 [MIT 许可证](LICENSE)。外部论文语料、模型和数据集不包含在仓库内，其许可单独适用。


