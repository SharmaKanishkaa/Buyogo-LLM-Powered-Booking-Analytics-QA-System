pip install langchain-community==0.0.31
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from typing import List, Dict, Any
import os

class HotelBookingRAG:
    def __init__(self, analytics_data: Dict[str, Any]):
        self.analytics = analytics_data
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_db = None
        self.qa_chain = None
        self._setup_rag_system()

    def _create_documents(self) -> List[Document]:
        documents = []

        documents.append(Document(
            page_content=(
                f"Booking Summary:\n"
                f"- Total bookings: {self.analytics['summary_stats']['total_bookings']:,}\n"
                f"- Cancellation rate: {self.analytics['summary_stats']['cancellation_rate']:.1%}\n"
                f"- Average lead time: {self.analytics['summary_stats']['avg_lead_time']:.1f} days"
            ),
            metadata={"category": "summary"}
        ))

        monthly_data = self.analytics['monthly_metrics']
        for month, adr in monthly_data['monthly_adr'].items():
            documents.append(Document(
                page_content=(
                    f"Month: {month}\n"
                    f"- Average Daily Rate: ${adr:.2f}\n"
                    f"- Total Revenue: ${monthly_data['monthly_revenue'].get(month, 0):,.0f}"
                ),
                metadata={"category": "monthly", "month": month}
            ))

        cancel_data = self.analytics['cancellation_analysis']
        documents.append(Document(
            page_content=(
                "Top Cancellation Rates by Country:\n" +
                "\n".join([f"- {country}: {rate:.1%}" 
                          for country, rate in cancel_data['by_country'].items()])
            ),
            metadata={"category": "cancellations"}
        ))

        documents.append(Document(
            page_content=(
                "Cancellation Rates by Lead Time:\n" +
                "\n".join([f"- {group}: {rate:.1%}" 
                          for group, rate in cancel_data['by_lead_time'].items()])
            ),
            metadata={"category": "cancellations"}
        ))

        return documents

    def _setup_rag_system(self) -> None:
        vectorstore_dir = os.path.join(os.getcwd(), "vectorstore", "hotel_rag")
        faiss_index_path = os.path.join(vectorstore_dir, "index.faiss")

        if os.path.exists(faiss_index_path):
            print("🔄 Loading vector store from disk...")
            self.vector_db = FAISS.load_local(
                folder_path=vectorstore_dir,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            print("📦 Creating vector store from analytics data...")
            documents = self._create_documents()
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
            os.makedirs(vectorstore_dir, exist_ok=True)
            self.vector_db.save_local(vectorstore_dir)
            print(f"✅ Vector store saved to {vectorstore_dir}")

        # Load LLM
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={
                "temperature": 0.3,
                "max_length": 512,
                "do_sample": True
            },
            huggingfacehub_api_token="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"
        )

        # Create custom prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful analytics assistant. Use the context below to answer the user's question as clearly and concisely as possible.

Context:
{context}

Question:
{question}

Answer:"""
        )

        # Build QA chain manually using prompt
        qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=prompt_template
        )

        self.qa_chain = RetrievalQA(
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            combine_documents_chain=qa_chain,
            return_source_documents=True
        )

    def query(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call _setup_rag_system() first.")

        result = self.qa_chain({"query": question})

        return {
            "answer": result["result"].strip(),
            "sources": [doc.page_content for doc in result["source_documents"]],
            "metadata": [doc.metadata for doc in result["source_documents"]]
        }

    def save_vector_db(self, path: str = "vectorstore/hotel_rag") -> None:
        if self.vector_db:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.vector_db.save_local(path)

    def load_vector_db(self, path: str = "vectorstore/hotel_rag") -> None:
        if os.path.exists(os.path.join(path, "index.faiss")):
            print("🔄 Loading vector store from disk...")
            self.vector_db = FAISS.load_local(
                folder_path=path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            self._setup_rag_system()
            print(f"✅ Vector store loaded from {path}")
        else:
            raise FileNotFoundError(f"Vectorstore not found at '{path}'")

    def get_available_contexts(self) -> List[Dict[str, Any]]:
        if not self.vector_db:
            return []
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in self.vector_db.similarity_search("", k=100)
        ]
