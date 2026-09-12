# Paper Review Workflow · 论文评审工作台

[English](./README_EN.md) | 简体中文

基于大模型的学术论文评审工作流引擎。上传一篇 PDF，选择目标会议（NeurIPS / ICML / ACL）与维度权重，由多维度 LLM 评审流水线产出结构化评分、综合评审意见与最终投稿建议。

## 功能特性

- **多会议评审**：NeurIPS（3 维度，1-10 分）、ICML（4 维度，1-4 分）、ACL（4 维度，1-4 分），各会议独立的维度、权重、分数区间与推荐阈值
- **多厂商大模型**：智谱 GLM、OpenAI、DeepSeek、Anthropic Claude，按次提供 API Key 即可，也支持环境变量配置
- **临时评审模式**：PDF + Key 仅在内存中使用，Key 可随时手动删除，任务可整体删除，不做持久化
- **实时结构化日志**：终端风格，展示模型调用参数、Token 用量、思考过程、各阶段状态流转
- **断点续跑**：任务失败后可从断点恢复，已成功的阶段不重复消耗 Token
- **OpenReview 导出**：评审结果一键导出为 OpenReview XML
- **双主题界面**：纸感学术评审台（浅色）/ 深空精密工作台（暗色），右下角一键切换

## 快速开始

```bash
git clone <repo>
cd paper-review-workflow
pip install -e .
python main.py server
```

浏览器打开 <http://localhost:8000/>，「发起评审」填写任务名称、选择会议与权重、上传 PDF、选择模型厂商与模型、粘贴 API Key，点击开始评审。

### 环境变量方式配置模型（可选）

不通过界面的任务也可以用环境变量指定模型：

```bash
export LLM_PROVIDER=zhipu          # zhipu / openai / deepseek / anthropic
export ZHIPU_API_KEY=...           # 各厂商对应的环境变量
# LLM_MODEL 缺省值：zhipu→glm-4.6，anthropic→claude-sonnet-4-6，其余需显式指定
```

## Web 界面

| 页面 | 功能 |
| --- | --- |
| 发起评审 | 任务名称、会议与维度权重、PDF 上传、厂商与模型、API Key |
| 监控 | Run 列表（按任务名展示）、Run 详情、Token 用量统计、实时日志、取消 / 续跑 / 删除 API-Key / 删除任务 |
| 决策 | 推荐结论、各维度分数、关键评估、决策依据、综合评审（可复制） |

主题切换：右下角日 / 月图标，选择持久化保存。

## CLI

```bash
python main.py server                                   # 启动 API + Web 服务
python main.py run configs/neurips_review.yaml \
    --payload '{"paper_source": "/path/to/paper.pdf"}'  # 运行评审
python main.py list-runs                                # 历史任务
python main.py show-run <run_id>                        # 任务详情
python main.py resume <run_id>                          # 从断点恢复
python main.py export <run_id> --format xml             # 导出 OpenReview XML
```

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/review` | 临时评审：multipart 上传 PDF + api_key + model + venue + weights + task_name |
| GET | `/api/runs` | 任务列表（支持 status 过滤） |
| GET | `/api/runs/{id}` | 任务详情（含步骤日志、Token 用量） |
| POST | `/api/runs/{id}/cancel` | 取消 |
| POST | `/api/runs/{id}/resume` | 续跑（可携带 api_key） |
| POST | `/api/runs/{id}/delete-key` | 删除内存中的 API Key |
| DELETE | `/api/runs/{id}` | 删除任务（Key、会话目录、运行记录一并清除） |
| GET | `/api/runs/{id}/decision` | 决策数据 |
| GET | `/api/runs/{id}/review` | 综合评审 Markdown |
| GET | `/api/runs/{id}/export` | OpenReview XML |
| GET | `/api/venues` | 会议配置 |
| WS | `/ws` | 全量实时事件 |

交互式文档：<http://localhost:8000/docs>

## 架构

复用 lwf 工作流引擎骨架（GitHub Actions 风格 YAML、三层状态机、矩阵并行），评审相关动作为：

- `extract`：PDF 解析（仅支持本地 PDF），产出元数据与全文
- `dim_score`：按会议维度矩阵并行打分（max-parallel: 2），结构化输出含证据引用
- `synthesize`：跨维度综合评审
- `decide`：纯规则加权评分，映射 OpenReview 七档推荐

运行目录结构：

```
sessions/<run_id>/
  upload/paper.pdf          # 上传的原始 PDF
  00_extract/               # 元数据、全文
  10_dim_<dimension>/       # 各维度 score.json + review.md
  50_synthesize/            # 综合评审
  60_decision/decision.json # 最终决策
  final_report.md
sessions/runs/<run_id>.json # 运行状态（JSON 文件存储，无数据库）
```

## 模型厂商支持

| 厂商 | 环境变量 | 默认模型 | 结构化输出 |
| --- | --- | --- | --- |
| 智谱 GLM | `ZHIPU_API_KEY` | glm-4.6 | json_object + schema 注入 |
| OpenAI | `OPENAI_API_KEY` | 需指定 `LLM_MODEL` | json_schema |
| DeepSeek | `DEEPSEEK_API_KEY` | 需指定 `LLM_MODEL` | json_object + schema 注入 |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 | tool use |

所有 provider 共享：3-5 次指数退避重试、schema 校验失败自动回传错误自愈重试、限流（429）长退避。

## 隐私与数据安全

- API Key 只保存在服务进程内存中，永不落盘（存储层对 `*_API_KEY` 字段脱敏兜底）
- 「删除 API-Key」立即清除内存中的 Key；「删除任务」连同 PDF、评审产物、运行记录一并删除
- 上传的 PDF 与评审产物默认保留，需要时手动删除或整任务删除

## 测试

```bash
pytest                    # 单元 + 集成（无需 API Key）
pytest --run-e2e -m e2e   # 端到端（需要真实 Key）
```

## License

MIT
