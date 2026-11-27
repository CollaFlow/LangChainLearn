# AGENTS.md

## Project overview

This workspace contains **RAG-Multi-Corpus**, a synthetic, multi-domain, multi-format dataset for evaluating Retrieval-Augmented Generation (RAG) systems, plus a small root-level learning project. `basic_rag.py` is a runnable single-document RAG example using TokenHub embeddings, FAISS retrieval, and TokenHub HY4 for answer generation. The bundled benchmark remains primarily **data and precomputed artifacts** rather than a reproducible application.

The corpus covers five fictional organizations: Aventro Motors, Cendara University, CloudWay 24, Velvera Technologies, and ZX Bank. Content is supplied as Markdown, HTML, DOCX, PDF, and PPTX. The main documentation and dataset text are in English. The dataset is licensed under the MIT License in `RAG-Multi-Corpus-main/LICENSE`.

## Repository layout

- `README.md`: Chinese setup, configuration, retrieval-only, full RAG, and test instructions.
- `basic_rag.py`: minimal synchronous Markdown RAG pipeline with separate loading, splitting, indexing, retrieval, context formatting, and answer-generation functions.
- `.env.example`: TokenHub API key, OpenAI-compatible base URL, HY4 chat model, and TokenHub embedding model settings.
- `tests/test_basic_rag.py`: standard-library `unittest` coverage using fake embeddings, with no API calls.
- `requirements.txt`: exactly pinned dependencies for the Python LangChain experimentation environment. It includes LangChain, LangGraph, OpenAI, FAISS, NumPy, Pydantic, SQLAlchemy, and related packages.
- `.venv/`: local virtual environment created with Python 3.11.9; it is not project source.
- `.gitignore`: ignores `.venv/`, `.env`, `__pycache__/`, and Python bytecode.
- `.gitattributes`: configures Git LFS for PDF, PPTX, DOCX, PNG, JPG/JPEG, GIF, and WebP files.
- `RAG-Multi-Corpus-main/README.md`: dataset purpose, organization overview, benchmark use cases, and citation.
- `RAG-Multi-Corpus-main/datasets/`: source corpora and query data.
- `RAG-Multi-Corpus-main/bechmark/bechmark-agentic-references/`: benchmark documentation, chunking artifacts, queries, and results. The `bechmark` spelling is part of the actual path and must not be silently corrected.
- `RAG-Multi-Corpus-main/metadata/`: organization JPG paths. These files are currently Git LFS pointer text rather than decoded images.

Each enterprise corpus has parallel `md`, `html`, `docx`, `pdf`, and `pptx` directories. Document stems generally align across formats, which supports cross-format comparisons. Logical document counts are 50 for Aventro Motors, 40 for Cendara University, 37 for CloudWay-24, 38 for Velvera Technologies, and 71 for ZX Bank: 236 logical documents, or 1,180 files across the five formats.

Important data files:

- `datasets/Dataset categories - queries_01122025.csv`: 1,252 query rows with columns `Enterprise Name`, `Query Type`, `Query`, and `Supporting Facts`. Supporting facts are JSON arrays containing `filename` and `text` objects.
- `datasets/Aventro Motors/Aventro Motors.csv`: generated query/fact responses by document; two document responses are empty.
- `bechmark/bechmark-agentic-references/Dataset categories - queries.csv`: 1,088 benchmark query rows; unlike the global CSV, it excludes the 164 Cloudway rows.

## Technology and intended benchmark architecture

The benchmark documentation describes Python with NumPy, Transformers, and PyTorch. It uses `BAAI/bge-m3` for 1,024-dimensional embeddings and GPT-4.1 for agentic chunking and relevance evaluation. Retrieval is based on cosine similarity, with Recall@k, Precision@k, MRR, and NDCG metrics.

The documented data flow is:

1. Read raw enterprise Markdown from `datasets/<enterprise>/md/`.
2. Produce agentic chunks with GPT-4.1.
3. Generate BGE-M3 embeddings for the chunks.
4. Load queries and supporting facts from CSV.
5. Retrieve top-k chunks using cosine similarity.
6. Judge retrieved results against supporting facts with GPT-4.1.
7. Calculate and aggregate retrieval metrics by query type, enterprise, and overall.
8. Save detailed JSON and aggregated CSV results.

Two chunk representations are checked in:

- `parsed-chunks/`: five JSON Lines files despite their `.json` suffix. Records include `file_name`, `title`, `chunks`, `usage`, and `time`.
- `parsed-chunks-less-tokens/`: five JSON arrays with a different, richer schema including `filename`, `markdown_content`, `structure_map_output`, `gpt_chunks`, `chunks`, `chunks_string`, `token_usage`, and `time_taken`.

The corresponding `benchmark_agentic_chunks_aggregated.{json,csv}` and `benchmark_agentic_chunks_less_tokens_aggregated.{json,csv}` files are precomputed evaluation outputs. Both stored result sets use `k=6`; they contain 787 and 786 evaluation records respectively. CloudWay is absent from these stored evaluation outputs.

**Reproducibility limitation:** the documentation refers to `search_bench.py`, but that script is absent. The chunk artifacts also contain text rather than the documented embedding vectors. Consequently, the benchmark cannot be rerun end to end from the checked-in files alone.

## Setup, build, and run commands

There is no package/build configuration (`pyproject.toml`, `setup.py`, `package.json`, lockfile, or Makefile) and no application build step.

Run the basic example after copying `.env.example` to `.env` and setting credentials:

```powershell
.venv\Scripts\python.exe basic_rag.py --retrieve-only
.venv\Scripts\python.exe basic_rag.py
```

The example creates a fresh in-memory FAISS index on every run. Retrieval-only mode still calls the configured embedding API; only chat generation is skipped.

To recreate the root Python dependency environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The benchmark README separately documents these additional dependencies, which are not all present in the root requirements file:

```powershell
.venv\Scripts\python.exe -m pip install numpy transformers torch
```

Use `git lfs pull` when LFS-backed binary files appear as pointer files. There is no valid benchmark run command while `search_bench.py` and the intermediate embeddings are missing; do not invent an entry point or claim that stored results were regenerated.

## Development conventions

- Preserve exact paths and filenames. Names include spaces, ampersands, Unicode punctuation, leading spaces in some artifact records, and existing misspellings. In particular, `CloudWay-24` is a directory name while query data uses `Cloudway 24`.
- Keep matching document stems aligned when adding or renaming cross-format content.
- Do not treat the two chunk directories as interchangeable: one uses JSON Lines and the other uses JSON arrays with different fields.
- Preserve dependency pinning style (`package==version`) when changing `requirements.txt`.
- `requirements.txt` is UTF-16 LE with BOM; preserve its encoding unless deliberately converting it. `.gitignore` was deliberately converted to UTF-8 because Git did not recognize the UTF-16 ignore rules.
- Binary documents are LFS-managed; do not edit them as plain text.
- `.DS_Store` files are present in the dataset and are not covered by the current `.gitignore`.
- No formatter, linter, editor configuration, pre-commit setup, or source-code style guide exists.

## Testing and validation

Run the basic example's offline tests with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests use fake embeddings and do not require credentials. There is no test-runner configuration or CI workflow. Benchmark JSON/CSV files are evaluation artifacts, not executable tests.

For data changes, validate the affected CSV/JSON schema and record counts, parse JSON Lines files line by line, and check cross-format filename alignment. Avoid casually normalizing text encodings: some existing artifacts contain mojibake or replacement characters, and some chunk filenames do not exactly match their source Markdown names.

Be aware of documentation/data discrepancies. The main README reports 786 queries, while the global and benchmark CSVs contain 1,252 and 1,088 rows respectively. The optimized Velvera chunk artifact also has an empty `chunks` list and null `gpt_chunks` for `promotional_offers.md`. Treat these as existing data characteristics unless a task explicitly asks to repair them.

## Security and deployment

All organizations and documents are synthetic. Passwords or security policies appearing inside documents are dataset subject matter, not application credentials. No real API keys are checked in. Recreating GPT-4.1-based processing or running the basic RAG example requires external credentials; keep them out of source and generated artifacts.

The basic example loads `TOKENHUB_API_KEY`, `TOKENHUB_BASE_URL`, `TOKENHUB_CHAT_MODEL`, and `TOKENHUB_EMBEDDING_MODEL` through `python-dotenv`; `.env` is ignored and only `.env.example` should be committed. Use TokenHub `hy4` for chat/LLM examples unless the user chooses otherwise. HY4 is not an embedding model: the default retrieval model is TokenHub `kinfra-text-embedding-0.6b` (1,024 dimensions), through the same OpenAI-compatible endpoint `https://tokenhub.tencentmaas.com/v1`.

There is no Dockerfile, Compose configuration, deployment manifest, hosting setup, release process, or deployment documentation. The project is a learning example plus a dataset/benchmark artifact repository, not a deployable service.

<!-- OPENWIKI:START -->

## OpenWiki

This repository has a generated `openwiki/` evidence index. It is optional just-in-time context, not required startup reading.

- Do not enumerate, preload, or search wikis at task start. Use retrieval when the user asks for it, when unfamiliar architecture or dependency behavior materially affects the task, or when source inspection leaves an important uncertainty. Stop once the question is grounded.
- When those conditions apply and OpenWiki retrieval tools are available, use `openwiki_search` for just-in-time context and `openwiki_read` for the relevant complete sections. If search returns `workspace_required`, ask which listed workspace to use and retry with its ID.
- Use `openwiki_list_workspaces` or `openwiki_list_wikis` when workspace membership itself needs to be discovered.
- If the retrieval tools are unavailable, read `openwiki/quickstart.md` and follow its links to the relevant pages.
- Treat source code and tests as authoritative. A brief's unknowns and review items are verification gaps, not automatic requirements.
- Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.

Use the repository's documented maintenance process to refresh generated pages after source changes. Check its configuration and completed runs before relying on automatic updates. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
