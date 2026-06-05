import fitz
import docx
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import uuid
import os

class RAGEngine:

    def __init__(self, groq_api_key):

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = Groq(
            api_key=groq_api_key
        )

        # Create local database folder
        os.makedirs("/tmp/chroma_db", exist_ok=True)

        # Persistent ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="/tmp/chroma_db"
        )

        self.collection = None

    # ----------------------------------------
    # Extract Text
    # ----------------------------------------

    def extract_text(self, file, file_type):

        text = ""

        if file_type == "pdf":

            pdf = fitz.open(
                stream=file.read(),
                filetype="pdf"
            )

            for page in pdf:
                text += page.get_text()

        elif file_type == "docx":

            document = docx.Document(file)

            text = "\n".join(
                [paragraph.text for paragraph in document.paragraphs]
            )

        elif file_type == "txt":

            text = file.read().decode(
                "utf-8",
                errors="ignore"
            )

        return text

    # ----------------------------------------
    # Chunk Text
    # ----------------------------------------

    def chunk_text(
        self,
        text,
        chunk_size=700,
        overlap=150
    ):

        chunks = []

        start = 0

        while start < len(text):

            chunks.append(
                text[start:start + chunk_size]
            )

            start += chunk_size - overlap

        return chunks

    # ----------------------------------------
    # Build Vector Index
    # ----------------------------------------

    def build_index(self, chunks):

        collection_name = (
            f"doc_{uuid.uuid4().hex[:8]}"
        )

        self.collection = (
            self.chroma_client.create_collection(
                name=collection_name
            )
        )

        embeddings = (
            self.embedding_model.encode(chunks)
        )

        self.collection.add(
            ids=[
                str(i)
                for i in range(len(chunks))
            ],
            embeddings=[
                embedding.tolist()
                for embedding in embeddings
            ],
            documents=chunks
        )

    # ----------------------------------------
    # Retrieve Documents
    # ----------------------------------------

    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedding_model.encode([query])
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )

        documents = results["documents"][0]

        distances = results["distances"][0]

        context = "\n\n".join(documents)

        return context, documents, distances

    # ----------------------------------------
    # General Chat
    # ----------------------------------------

    def general_chat(self, query):

        response = (
            self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",

                messages=[
                    {
                        "role": "system",
                        "content": """
You are an expert AI assistant.

Provide detailed answers.

Use:
- Headings
- Bullet points
- Examples
- Clear explanations

Answer in a teaching style.
"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],

                temperature=0.5,
                max_tokens=1500
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    # ----------------------------------------
    # RAG Chat
    # ----------------------------------------

    def rag_chat(self, query):

        context, docs, distances = self.retrieve(
            query
        )

        prompt = f"""
You are an expert AI assistant.

Use ONLY the supplied document context.

Provide a detailed answer.

Structure your answer as:

1. Overview
2. Detailed Explanation
3. Key Points
4. Conclusion

If the answer is not present in the document,
respond with:

I could not find that information in the uploaded documents.

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        response = (
            self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3,
                max_tokens=1500
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    # ----------------------------------------
    # Main Ask Function
    # ----------------------------------------

    def ask(
        self,
        query,
        mode="Auto"
    ):

        # General Chat Mode

        if mode == "General Chat":

            return self.general_chat(query)

        # Document Only Mode

        if mode == "Document Only":

            if self.collection is None:

                return (
                    "Please upload and process documents first."
                )

            return self.rag_chat(query)

        # Auto Mode

        if self.collection is None:

            return self.general_chat(query)

        try:

            context, docs, distances = (
                self.retrieve(query)
            )

            best_distance = min(distances)

            # Weak retrieval -> use LLM

            if best_distance > 1.2:

                return self.general_chat(query)

            answer = self.rag_chat(query)

            if (
                "could not find" in answer.lower()
                or
                "not found" in answer.lower()
            ):

                return self.general_chat(query)

            return answer

        except Exception as e:

            print(e)

            return self.general_chat(query)

