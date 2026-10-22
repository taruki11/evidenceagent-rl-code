# EvidenceAgent-RL

> 代码快照发布说明：仅含筛选源码，不含模型、数据、日志或凭据。

EvidenceAgent-RL 是一个围绕论文证据问答的 Agentic-RL 研究工程，研究如何把证据边界、检索策略记忆和工具交互纳入可审计的训练与评测流程。

## 离线快速开始

需要 Python 3.11 或更高版本，在独立虚拟环境中执行。包本身没有运行时依赖，
安装成功后运行不需要任何第三方库：

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

只有当"正例成功、跨 episode 作答被拦截、重放状态一致"三者同时成立时，示例才返回 0。
不带 `--output` 时不写任何文件；带 `--output` 时也绝不覆盖：路径已存在则退出码 2，
原文件字节不变。

## 你可以用这个示例核验什么

五类行为，全部离线且确定性：

1. 证据句柄通过引用校验后，受支持的断言被接受。
2. 未校验、伪造或 DOI 不在白名单内的句柄不能支撑作答。
3. 证据句柄不能跨 episode 复用，检索被限制在任务的 corpus scope 内。
4. 工具预算不会变成负数，被拒绝的动作不扣预算。
5. 序列化后的动作轨迹可以重放出完全一致的状态。

`tests/test_harness_contract.py` 独立于示例覆盖同样五类边界，任一侧回归都不会被另一侧掩盖。

## 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'evidence_agent_rl'` | 当前解释器没有安装本包。 | 在仓库根目录执行 `python -m pip install -e .`；不安装时把 `PYTHONPATH` 指向 `src`。 |
| `python -m unittest discover` 输出 `NO TESTS RAN` | 发现起点是当前目录，此处不会自动进入 `tests/`。 | 使用上文命令 `python -m unittest discover -s tests -v`。 |
| 安装前 `test_installed_module_cli_emits_json` 失败 | 该用例会启动子进程导入**已安装**的包。 | 先安装。测试集针对已安装的包，而非裸 checkout。 |
| `pip install` 卡在 "Installing build dependencies" 后失败 | 构建阶段需要联网拉取 `setuptools>=77`。 | 离线机器先装好 `setuptools>=77`，再执行 `python -m pip install --no-build-isolation -e .`。 |
| `pip install` 报 `project.license` 或 "License classifiers have been superseded" | 低于 77 的 `setuptools` 不认识 SPDX 许可证元数据。 | 升级 `setuptools`；该问题只在关闭构建隔离时出现。 |
| `demo.py: error: Cannot create output file: ... File exists` | `--output` 按设计不覆盖。 | 换一个新路径。退出码 2，原文件不变。 |

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


