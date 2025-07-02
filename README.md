# AI-Powered Internal Audit Assistant


*A sample GIF demonstrating the chatbot's functionality. You should create your own GIF and replace this link.*

## 📌 Overview

The **AI-Powered Internal Audit Assistant** is a specialized chatbot designed to enhance the productivity of internal audit teams. It leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide quick, accurate, and context-aware answers from a private collection of audit documents (PDFs, Word docs, Excel sheets, etc.).

This tool was built to address the common challenge auditors face: manually sifting through vast amounts of documentation to find specific policies, procedures, or checklist templates. By simply uploading their documents, auditors can interact with a smart assistant that understands their knowledge base, saving significant time and effort.

This project serves as a portfolio piece demonstrating the practical application of Generative AI to solve real-world business problems without requiring complex ERP systems or databases.

## ✨ Features

*   **Simple Web Interface**: An intuitive and user-friendly UI built with Streamlit.
*   **Multi-Format Document Upload**: Supports various document types including `.pdf`, `.docx`, `.xlsx`, and `.txt`.
*   **Intelligent Q&A**: Ask complex questions in natural language and receive answers sourced directly from your documents.
*   **Document Retrieval**: Instantly find and retrieve specific templates, checklists, or policies (e.g., "Find the IT security audit checklist").
*   **On-the-Fly Summarization**: Request summaries of lengthy documents to quickly grasp key points.
*   **Conversational Memory**: The chatbot remembers the context of the current conversation for a more natural follow-up.

## 🛠️ Tech Stack & Architecture

This project is built on a modern, Python-based AI stack. The core of the application is a **Retrieval-Augmented Generation (RAG)** pipeline.

1.  **Load & Chunk**: Documents are loaded and split into smaller, manageable chunks.
2.  **Embed (Index)**: Each chunk is converted into a numerical vector representation (embedding) using a local, open-source model (`HuggingFaceInstructEmbeddings`). This process is **100% private and free**.
3.  **Store**: The embeddings are stored in-memory in a `ChromaDB` vector store for efficient similarity searching.
4.  **Retrieve**: When a user asks a question, the system finds the most relevant document chunks from the vector store.
5.  **Generate**: The user's question and the retrieved chunks are passed to a Large Language Model (LLM) like OpenAI's GPT, which generates a coherent, human-like answer based on the provided context.

**Technologies Used:**
*   **Backend & Logic**: Python
*   **AI/LLM Framework**: [LangChain](https://www.langchain.com/)
*   **Web Framework**: [Streamlit](https://streamlit.io/)
*   **LLM for Generation**: [OpenAI GPT-3.5/4](https://platform.openai.com/) (Can be swapped with other models)
*   **Embedding Model**: [Hugging Face Sentence Transformers](https://huggingface.co/hkunlp/instructor-large) (Local & private)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/) (In-memory)
*   **Document Loaders**: `PyPDF`, `python-docx`, `openpyxl`

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.8+
*   An OpenAI API Key (only for the final answer generation step).

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AuditGPT-RAG.git
cd AuditGPT-RAG
```

### 2. Install Dependencies

Create a virtual environment and install the required Python packages.

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a file named `.env` in the root directory of the project and add your OpenAI API key:

```
OPENAI_API_KEY="sk-YourSecretKeyHere"
```

### 4. Run the Application

Launch the Streamlit app with the following command:

```bash
streamlit run audit_chatbot.py
```

Your web browser will automatically open to the application's URL (e.g., `http://localhost:8501`).


## 5. Screenshots

![alt text](/demo/250702_12h01m16s_screenshot.png)

## 💡 Future Improvements & Scalability

This project serves as a strong foundation. Potential future enhancements include:
*   **Persistent Vector Store**: Save the ChromaDB index to disk to avoid re-processing documents on every run.
*   **User Authentication**: Add a login layer to manage access for different users or teams.

---

Feel free to reach out if you have any questions or would like to discuss a custom implementation for your organization!