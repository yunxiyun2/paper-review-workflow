# OpenReview XML Export (Phase 2 #4) — SPEC-PRD

**版本**: 1.0
**日期**: 2026-08-16
**Phase**: 2 (subsystem #4 of 7)

---

## 1. 定位与范围

为 paper-review-workflow 增加 OpenReview XML 导出能力,将评审结果转为标准 OpenReview note XML 格式,支持 CLI 和 API 两种触发方式。

### 范围
- 新增 `paper_review_workflow/exporters/openreview.py` — OpenReview XML 生成器
- 改造 `paper_review_workflow/cli.py` — 加 `export` 子命令
- 改造 `paper_review_workflow/api/server.py` — 加 `GET /api/runs/{id}/export` endpoint
- 单元 + 集成测试

### 不做
- 其他导出格式(JSON 已有 via `show-run`)
- OpenReview API 上传(只导出 XML,不自动上传到 OpenReview 系统)
- 批量导出(Phase 2 #6)

---

## 2. OpenReview XML 格式

```xml
<?xml version="1.0" encoding="UTF-8"?>
<note>
  <content>
    <field name="recommendation">weak_accept</field>
    <field name="confidence">3</field>
    <field name="review">This paper proposes... [review.md 全文]</field>
    <field name="soundness">7</field>
    <field name="contribution">8</field>
  </content>
  <metadata>
    <venue>neurips</venue>
    <paper_id>a1b2c3d4</paper_id>
    <run_id>20260816-143022-a1b2c3d4</run_id>
  </metadata>
</note>
```

### 字段映射

| XML 字段 | 数据源 | 映射逻辑 |
|---|---|---|
| `recommendation` | `decision.json.recommendation` | 直接取(strong_accept 等) |
| `confidence` | `decision.per_dimension` 各维度 confidence | 加权平均 → 1-5 整数(round(avg × 5)) |
| `review` | `50_synthesize/review.md` | 全文 |
| `soundness` | `per_dimension["soundness"].score` | 所有 venue 都有 soundness |
| `contribution` | `per_dimension` | NeurIPS→contribution, ICML→significance, ACL→overall |

---

## 3. 架构

```
paper_review_workflow/
├── exporters/                    # ★ 新增
│   ├── __init__.py
│   └── openreview.py             # OpenReviewExporter 类
├── api/server.py                 # 改:加 GET /api/runs/{id}/export
├── cli.py                        # 改:加 export 子命令
└── tests/
    ├── unit/test_openreview_exporter.py
    └── integration/test_api_export.py
```

### OpenReviewExporter

```python
class OpenReviewExporter:
    def export(self, run_id: str, storage: StorageBackend) -> str:
        run = storage.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        decision = self._load_decision(run)
        review_md = self._load_review_md(run)
        # ... 构建 XML ...
```

---

## 4. CLI + API

### CLI
```bash
python main.py export <run_id> [--format xml] [--output <path>]
```

### API
```
GET /api/runs/{run_id}/export?format=xml
→ 200, Content-Type: application/xml, body=XML
→ 404 if run not found
```

---

## 5. 实施阶段

| 里程碑 | 范围 | 预估 |
|---|---|---|
| M1 | `exporters/openreview.py` + 单元测试 | 0.5 天 |
| M2 | CLI export 子命令 + API endpoint + 集成测试 | 0.5 天 |
| M3 | README + tag v0.6.0 | 0.5 天 |

**总预估: ~1.5 天**

---

## 6. 验收标准

- [ ] `OpenReviewExporter.export()` 返回 XML 字符串
- [ ] XML 含 5 字段 + metadata
- [ ] CLI `export` 子命令工作
- [ ] API `GET /api/runs/{id}/export` 返回 XML
- [ ] 404 for nonexistent
- [ ] 测试通过
- [ ] tag v0.6.0
