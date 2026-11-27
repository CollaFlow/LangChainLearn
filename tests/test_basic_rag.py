import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from basic_rag import (
    DEFAULT_DOCUMENT,
    build_vector_store,
    format_context,
    generate_answer,
    load_markdown,
    retrieve_documents,
    split_document,
)


class BasicRagTests(unittest.TestCase):
    def test_load_split_and_retrieve_without_api(self):
        document = load_markdown(DEFAULT_DOCUMENT)
        self.assertIn("Cheque Book", document.page_content)
        self.assertEqual(Path(document.metadata["source"]), DEFAULT_DOCUMENT)

        chunks = split_document(document, chunk_size=300, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("start_index" in chunk.metadata for chunk in chunks))

        vector_store = build_vector_store(chunks, FakeEmbeddings(size=16))
        results = retrieve_documents(vector_store, "cheque book", k=2)
        self.assertEqual(len(results), 2)

    def test_markdown_split_preserves_sections_and_offsets(self):
        text = (
            "# Guide\n\nIntroduction.\n\n"
            "## Mobile App\n\nOpen the app and submit a request.\n\n"
            "## Branch\n\nVisit a branch and bring your ID.\n"
        )
        document = Document(
            page_content=text, metadata={"source": "guide.md"}
        )

        chunks = split_document(document, strategy="markdown")
        app_chunk = next(
            chunk for chunk in chunks if "Open the app" in chunk.page_content
        )
        branch_chunk = next(
            chunk for chunk in chunks if "Visit a branch" in chunk.page_content
        )

        self.assertEqual(app_chunk.metadata["section"], "Guide > Mobile App")
        self.assertTrue(app_chunk.page_content.startswith("[Guide > Mobile App]"))
        self.assertNotIn("Visit a branch", app_chunk.page_content)
        self.assertEqual(branch_chunk.metadata["section"], "Guide > Branch")
        self.assertEqual(app_chunk.metadata["source"], "guide.md")
        self.assertTrue(
            text[app_chunk.metadata["start_index"] :].startswith("Open the app")
        )

    def test_markdown_split_recursively_splits_long_sections(self):
        text = "# Guide\n\n## Steps\n\n" + (
            "Complete every step before continuing to the next paragraph.\n\n" * 4
        )
        document = Document(page_content=text, metadata={"source": "guide.md"})

        chunks = split_document(
            document, chunk_size=100, chunk_overlap=0, strategy="markdown"
        )

        step_chunks = [
            chunk for chunk in chunks if "Complete every step" in chunk.page_content
        ]
        self.assertGreater(len(step_chunks), 1)
        self.assertTrue(
            all(
                chunk.metadata["section"] == "Guide > Steps"
                for chunk in step_chunks
            )
        )
        self.assertTrue(
            all(
                chunk.page_content.startswith("[Guide > Steps]")
                for chunk in step_chunks
            )
        )

    def test_context_contains_source_and_text(self):
        document = load_markdown(DEFAULT_DOCUMENT)
        context = format_context([document])
        self.assertIn("Apply for a Cheque Book.md", context)
        self.assertIn("3–7 working days", context)

    def test_generate_answer_with_fake_chat_model(self):
        document = load_markdown(DEFAULT_DOCUMENT)
        llm = GenericFakeChatModel(messages=iter(["根据资料，通常需要 3–7 个工作日。"]))
        answer = generate_answer("多久送达？", [document], llm)
        self.assertEqual(answer, "根据资料，通常需要 3–7 个工作日。")

    def test_missing_document_has_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            load_markdown(Path("not-found.md"))


if __name__ == "__main__":
    unittest.main()
