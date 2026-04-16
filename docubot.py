"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Builds an inverted index mapping lowercase tokens to the filenames
        that contain them.

        Structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Each filename appears at most once per token (deduped via seen set).
        Basic punctuation is stripped so "auth," and "auth" map to the same key.
        """
        index = {}
        punct = str.maketrans("", "", ".,!?;:\"'()[]{}")

        for filename, text in documents:
            seen = set()
            for word in text.lower().split():
                token = word.translate(punct)
                if not token or token in seen:
                    continue
                seen.add(token)
                if token not in index:
                    index[token] = []
                index[token].append(filename)

        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Returns a numeric relevance score: the total number of times
        any query word appears in the document text (term frequency sum).

        - Lowercases both query and text before comparing
        - Strips basic punctuation from text tokens
        - Different queries produce different scores because counts vary per doc
        """
        punct = str.maketrans("", "", ".,!?;:\"'()[]{}")
        query_tokens = [w.translate(punct) for w in query.lower().split()]
        text_tokens = [w.translate(punct) for w in text.lower().split()]

        score = 0
        for token in query_tokens:
            if token:
                score += text_tokens.count(token)
        return score

    def retrieve(self, query, top_k=3):
        """
        Uses the index to find candidate documents that contain at least one
        query word, scores each candidate, and returns the top_k results
        sorted by score descending as a list of (filename, text).
        """
        punct = str.maketrans("", "", ".,!?;:\"'()[]{}")
        query_tokens = [w.translate(punct) for w in query.lower().split() if w]

        # Use the index to narrow candidates (avoid scoring every document)
        candidate_filenames = set()
        for token in query_tokens:
            for filename in self.index.get(token, []):
                candidate_filenames.add(filename)

        scored = []
        for filename, text in self.documents:
            if filename not in candidate_filenames:
                continue
            score = self.score_document(query, text)
            if score > 0:
                scored.append((score, filename, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(filename, text) for _, filename, text in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
