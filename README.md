# LangChainLearn

这是一个从零学习 RAG（检索增强生成）的练习仓库。当前第一个示例使用素材库中的一篇 Markdown，完成：

```text
读取文档 → 文本切分 → Embedding → FAISS 向量索引
         → 相似度检索 → 将片段交给 LLM → 生成回答
```

`RAG-Multi-Corpus-main/` 是原始数据素材，不是本示例的应用代码。

## 1. 准备环境

需要 Python 3.11。仓库已有 `.venv` 时可直接使用；否则重新创建：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本示例使用 TokenHub 的 **HY4** 生成回答，并使用 TokenHub 文本向量模型完成检索。LangChain 的 OpenAI 兼容客户端依赖已经包含在 `requirements.txt` 中。

## 2. 配置模型

复制环境变量样例：

```powershell
copy .env.example .env
```

在 TokenHub 控制台创建 API Key，并确保 Key 可以调用 `hy4` 和 `kinfra-text-embedding-0.6b`。编辑 `.env`：

```dotenv
TOKENHUB_API_KEY=你的TokenHub真实密钥
TOKENHUB_BASE_URL=https://tokenhub.tencentmaas.com/v1
TOKENHUB_CHAT_MODEL=hy4
TOKENHUB_EMBEDDING_MODEL=kinfra-text-embedding-0.6b
```

模型分工如下：

- `hy4`：聊天模型，负责根据检索资料生成回答。
- `kinfra-text-embedding-0.6b`：文本向量模型，负责文档和问题向量化，输出 1024 维向量，支持中英文。

HY4 默认开启深度思考，回答可能需要等待一段时间。真实密钥只放在 `.env`，不要写入代码或提交 Git。

## 3. 先只运行检索

```powershell
.venv\Scripts\python.exe basic_rag.py --retrieve-only
```

这一步仍会调用 TokenHub Embedding API，但不会调用 HY4。控制台会显示：

1. 读取的文档路径。
2. 文档被切分成的 Chunk 数量。
3. 与默认问题最相似的 3 个片段及其来源。

默认问题是“如何申请支票簿？通常需要多久送达？”。先检查检索片段是否包含申请方式和 3–7 个工作日，再进入生成阶段。

## 4. 运行检索与回答

```powershell
.venv\Scripts\python.exe basic_rag.py
```

程序会把检索片段填入 Prompt，要求模型仅根据资料回答，并使用问题的语言输出。每次运行都会重新生成向量并创建内存索引，不会保存到磁盘。

也可以指定问题和参数：

```powershell
.venv\Scripts\python.exe basic_rag.py --question "通过网银申请支票簿有哪些步骤？" --k 2
.venv\Scripts\python.exe basic_rag.py --chunk-size 300 --chunk-overlap 50 --retrieve-only

# 对比 Markdown 标题感知切分
.venv\Scripts\python.exe basic_rag.py --chunk-strategy markdown --retrieve-only
```

`--chunk-strategy recursive` 是默认的通用递归切分；`markdown` 会先按标题划分章节，再递归控制 Chunk 长度。后者把标题路径写入 `section` metadata 和 Chunk 文本，避免跨章节混合，让标题语义参与向量检索。

## 5. 对照代码学习

`basic_rag.py` 中的函数与 RAG 环节一一对应：

| 函数 | 作用 |
|---|---|
| `load_markdown` | 读取文本，用 `Document` 保存内容和来源 |
| `markdown_sections` | 按 Markdown 标题划分章节并生成标题路径 |
| `split_document` | 使用递归或 Markdown 策略切分，保留来源及原文位置 |
| `build_vector_store` | 为 Chunk 生成向量并建立 FAISS 索引 |
| `retrieve_documents` | 按相似度找出与问题最接近的 k 个 Chunk |
| `format_context` | 将检索结果整理为模型上下文 |
| `generate_answer` | 用 `Prompt → LLM → 输出解析器` 生成回答 |

建议按以下顺序做小实验，每次只改一个变量：

1. 修改 `--k`，观察返回片段数量。
2. 修改 `--chunk-size` 和 `--chunk-overlap`，观察 Chunk 边界。
3. 对比 `--chunk-strategy recursive` 与 `markdown`，检查标题路径和步骤完整性。
4. 问一个文档没有答案的问题，检查模型是否承认资料不足。
5. 换一篇素材库 Markdown，使用 `--document` 指定路径。

## 6. 离线测试

测试使用假 Embedding，不需要 API Key，也不会产生模型费用：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前示例是学习用的同步、单文档、内存版 RAG；多文档加载、索引持久化、Rerank 和系统评测留待后续阶段。
