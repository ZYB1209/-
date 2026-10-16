## 在线标注网页

[打开 MFL Lab](https://zyb1209.github.io/-/)：浏览器内导入三轴CSV、框选采样点与通道范围、导入导出标注。数据在浏览器内处理，无自动同步；原Python智能体检测需另外部署后端。

# 钢丝绳三轴漏磁智能体 v3

研究原型：通过数值工况特征和可选 Qwen 视觉观察识别噪声，由 DeepSeek 调度受约束的信号处理工具，执行自适应滤波、异常区间检测、验证及交互解释。

本仓库只包含代码、专家知识卡与说明，不包含采集数据、真实缺陷标注、API 密钥、模型缓存或运行结果。数据需要在本机单独准备。

## 工作流

1. 读取三轴、每轴 32 通道的漏磁矩阵，检查数据与坐标条件。
2. 提取数值工况特征；将三轴信号渲染为固定规则的图像供视觉模型观察。
3. 合并数值和视觉证据，识别股纹、抖动等噪声及局部异常响应。
4. 调度器从 A/B/C/D 工具组合选择处理方案；参数守卫限制工具和参数范围。
5. 执行滤波，验证响应保留、位移与背景变化；失败时最多重规划一次，再回退到基线。
6. 检测时间候选并复核响应范围，输出采样点、名义时间、证据图和工具调用记录。
7. 用户可继续追问候选及比较流程；真实标注仅供独立评分读取。

工况与验证采用工程启发式指标，尚非经过统计校准的置信度。输出区间是漏磁响应范围；米制位置需要同步实测位置表，不能直接视为物理断口长度。

## 安装与启动

Python 3.10+。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

浏览器访问 `http://127.0.0.1:8767`。Windows 也可在准备好 Python 和依赖后双击 `start.cmd`。启动脚本会优先使用本机 Codex 附带 Python，找不到时使用 PATH 中的 Python；虚拟环境用户建议使用上面的显式命令。

Linux/macOS 可使用 `python3 -m venv .venv`，再通过 `.venv/bin/python` 安装依赖和运行 `app.py`。服务默认绑定本机回环地址；Codespaces 中网页转发尚未适配或验证，本仓库可用于编辑代码，运行建议先在本机进行。

## 数据准备

请参阅 [数据格式](dataset/README.md)。将原数据集单独解压到 `dataset/MFL_TriAxial_AllAngles_Dataset/`，至少包含 `manifest.csv` 和它引用的 `data/` 文件；评估另需对应 `annotations/` 文件。不提供数据时不能进行真实记录检测，相关单元测试仍可独立运行。

## 模型配置

Windows：在网页分别填写调度 API 和视觉 API 的 Base URL、模型名、密钥，密钥由当前用户的 Windows DPAPI 加密保存。配置文件禁止提交。

其他系统或服务器：在进程环境中配置，不将密钥写入仓库。

| 用途 | 环境变量 |
|---|---|
| 调度 API | `MFL_API_BASE`、`MFL_MODEL`、`MFL_API_KEY` |
| 视觉 API | `MFL_VLM_BASE`、`MFL_VLM_MODEL`、`MFL_VLM_KEY` |
| 可选本地 Ollama | `MFL_OLLAMA_MODEL` |

Base URL 和模型名以你购买的服务实际提供值为准，模型需分别支持工具调用或视觉输入。项目不会自动读取 `.env`；可通过操作系统或托管平台的环境变量/Secrets 功能注入。不要在聊天、README 或 Git 提交中放入密钥。

## 评估

```powershell
python evaluate_agent.py
python evaluate_agent.py --backend rules --mode fast --perception numeric
python evaluate_agent.py --resume runs/evaluations/你的评估目录
python view_evaluation.py runs/evaluations/你的评估目录
```

默认对 19 条实际采集记录调用已配置的调度/视觉 API，可能产生服务费用；派生角度视图默认不当作独立样本。每条先存预测，再读取真实标注，按整条记录、一对一最大基数匹配计算区间精确率、召回率、F1，主阈值 IoU≥0.5。额外提供 IoU≥0.3/0.7 和采样点混淆矩阵。正常点较多时不能仅看普通准确率。

结果保存到被 Git 忽略的 `runs/evaluations/`，含报告、逐条表格、误报漏检清单和预测快照。`--resume` 在冻结配置不变时复用已完成预测，避免重复调用。真实标注索引暂按 0 起始闭区间解释，需数据作者确认。现有数据曾用于开发，不能直接当作独立测试集。

## 无需数据与真实 API 的检查

```powershell
python -m unittest test_evaluate_agent
```

其余 `test_*.py` 覆盖滤波、工具调度和接口；部分需要本地数据或启动服务，勿把全部测试默认理解为不需数据。

## 主要模块

| 模块 | 功能 |
|---|---|
| `app.py` / `index.html` | 本地服务与交互界面 |
| `core.py` / `workflow.py` | 数据读取、检测与工作流 |
| `cloud_agent.py` / `ollama_agent.py` | 云端与本地调度 |
| `visual_condition.py` | 三轴图像、视觉 API 与结构校验 |
| `adaptive_signal.py` / `agentic_dsp.py` | 工况特征与自适应滤波流程 |
| `dsp_tools.py` / `parameter_guard.py` / `verifier.py` | 处理工具、参数守卫与结果验证 |
| `expert_knowledge.py` / `knowledge/` | 有出处的钢绳领域知识卡 |
| `evaluate_agent.py` / `view_evaluation.py` | 真实标注评分与可视化 |

算法属于参考论文的工程实现及启发式扩展，尚未证明智能体优于固定滤波检测流程，也未训练新的图像检测器。科研贡献和泛化性能需要独立数据、对照及消融实验验证。

## 在线标注与检测平台（待云端部署）

新增[web-platform](web-platform/README.md)：私有访问、三轴CSV上传、时间与通道范围框选、标注版本、后台检测及冻结预测评分。源代码与Docker配置已准备；实际访问地址、持久磁盘与模型Secrets需要在目标云账号完成配置。原始数据和密钥不进入GitHub。
