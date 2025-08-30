from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI()

# -------------------------
# Load documents
# -------------------------
loader = TextLoader("content.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# -------------------------
# Create embeddings and vectorstore
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# -------------------------
# Load LLM using Transformers pipeline
# -------------------------
model_name = "EleutherAI/pythia-410m"  # Public, no authentication required
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

text_gen_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # GPU or -1 for CPU
    max_new_tokens=200
)


# -------------------------
# Wrapper function for RAG chain
# -------------------------
class PipelineWrapper:
    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, prompt):
        output = self.pipe(prompt)
        return output[0]['generated_text']

llm = PipelineWrapper(text_gen_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# -------------------------
# FastAPI endpoint
# -------------------------
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = qa.run(query.question)
    return {"answer": answer}
