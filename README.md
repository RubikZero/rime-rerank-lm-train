# rime-rerank-lm-train

lm_ranker 插件的模型训练、数据准备、ONNX 导出，以及配套的评测与隔离测试工具。

- 插件与推理代码：https://github.com/RubikZero/rime-lm-ranker
- librime fork（含插件接口）：https://github.com/RubikZero/librime
- Windows 前端与安装包：https://github.com/RubikZero/weasel

## 目录

| 路径 | 内容 |
|---|---|
| `*.py`（根目录） | 数据准备、训练（MLM 微调 / reranker）、评测、ONNX 导出 |
| `onnx/mlm_ft_v2/` | 线上模型的配置与词表（`lm_config.json`、`vocab.txt`） |
| `tools/` | 隔离运行时 harness、IPC 客户端源码与构建脚本、S0/S1 探针、生产环境校验脚本 |

## 数据与权重

训练语料、检查点与导出的 ONNX 权重体积很大，**不放在 git 里**：语料与检查点原由 DVC 管理，
权重与安装包随 Release 发布。脚本中的路径常量（`RIME_ROOT` 之类）需按本机环境调整。

## 流程概览

1. 语料 → `prepare_data.py` / `prepare_reranker_requests*.py` / `build_reranker_npz.py`
2. 训练 → `train_mlm.py`（掩码语言模型微调）、`train_mlm_reranker.py`、`train_reranker.py`
3. 导出 → `export_onnx.py`（同时产出 `encode_prefix.onnx`、`score_tail.onnx` 两张上下文缓存图）
4. 评测 → `eval_*.py`、`tools/s1_verify_graphs.py`、`tools/s0_prefix_cache_probe.py`
5. 实机验证 → `tools/s1_*.ps1`、`tools/post_install_check.ps1`（直连已安装的 WeaselServer）

## 许可

脚本以 BSD 3-Clause 发布（见 `LICENSE`）。基座模型 `uer/chinese_roberta_L-4_H-256`、
ONNX Runtime 及训练语料的许可请见各自来源与 Release 说明。