# 测试

```bash
# 单元测试（不需要 Milvus / 模型权重）
uv run pytest -m "not contract and not integration" -q

# 跑全部测试（contract 测试需要真实 Milvus ≥ 2.4）
uv run pytest -q
```

## 测试组织

- `tests/unit/`：纯单元测试，使用 fake / mock embedder，秒级可跑完。
- `tests/contract/`：contract 测试，需要真实 Milvus server，通过 `@pytest.mark.contract` 标记。

合约测试示例：连接 `VS_MILVUS_URI` 上的真实 Milvus，验证 database/collection/upsert/search 端到端可用。
