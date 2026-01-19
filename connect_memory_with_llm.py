import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

# STEP 1: Load environment variables
load_dotenv()

os.environ.setdefault(
    "GROQ_API_KEY",
    os.getenv(
        "GROQ_API_KEY",
        "GROQ_API_KEY"
    ),
)

# STEP 2: Load LLM
def load_llm(model_name="llama-3.3-70b-versatile"):
    return ChatGroq(
        model=model_name,
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

# STEP 3: Custom Legal Prompt 
CUSTOM_PROMPT_TEMPLATE = """"
You are a compassionate mental health support assistant.
Your role is to provide brief, calm, and supportive mental health guidance.

Detect the user's language and respond in the same language.
Use the provided context only if it is relevant to mental health support.
If the context is insufficient, say "I don't know" gently.

RESPONSE RULES:
- Keep responses short (3–5 sentences maximum).
- Use simple, reassuring language.
- Focus on emotional validation and one practical coping suggestion.
- Do not provide medical diagnoses or prescribe medication.
- Do not replace a mental health professional.
- Avoid long explanations or detailed lists.
- If the user expresses severe distress or self-harm thoughts, briefly encourage seeking professional or trusted help.

User message:
{question}

Mental health context:
{context}

Brief supportive response:
"""


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# STEP 4: Embeddings & FAISS Index
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
)

DB_FAISS_PATH = "vectorstore/db_faiss"
LEGAL_DOCS_PATH = "data/"  

def build_faiss_index():
    print("Building FAISS index from legal documents...")
    
    all_documents = []
    
    folder_path = Path(LEGAL_DOCS_PATH)
    if not folder_path.exists():
        raise ValueError(f"Folder {LEGAL_DOCS_PATH} does not exist. Create it and add your legal files.")
    
    for file_path in folder_path.iterdir():
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = UnstructuredPDFLoader(str(file_path))
            elif file_path.suffix.lower() in [".txt"]:
                loader = TextLoader(str(file_path))
            elif file_path.suffix.lower() in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                print(f"Skipping unsupported file: {file_path}")
                continue
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    if not all_documents:
        raise ValueError("No legal documents were loaded. Add PDFs/TXT/DOCX in 'legal_docs/' folder.")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)
    
    # Build FAISS
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS index built successfully.")
    return db

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception:
    print("FAISS index not found or embedding mismatch. Rebuilding...")
    db = build_faiss_index()

# STEP 5: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm("llama-3.3-70b-versatile"),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=False, 
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# STEP 6: Run chatbot
if __name__ == "__main__":
    print("Legal Chatbot (type 'exit' to quit)")
    
    while True:
        user_query = input("\nEnter your legal query: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        try:
            response = qa_chain.invoke({'query': user_query})
            print("\nANSWER:\n", response["result"])
        except AssertionError:
            print("Embedding dimension mismatch detected. Rebuilding FAISS index...")
            db = build_faiss_index()
            qa_chain.retriever = db.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            print("Error:", e)