import os
from dotenv import load_dotenv
from huggingface_hub import login
import pydantic
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

# Configure Pydantic to avoid conflicts with protected namespaces
pydantic.BaseConfig.protected_namespaces = ()

# Set environment variables
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load environment variables from .env file
load_dotenv()

# Authenticate with Hugging Face
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
if huggingface_api_key:
    login(token=huggingface_api_key)
else:
    raise ValueError("HUGGINGFACE_API_KEY is not set in the .env file")

app = Flask(__name__)

# Load the model and tokenizer
try:
    model_id = "meta-llama/Llama-2-7b-chat-hf"  # Change if needed
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load model with CPU only
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", low_cpu_mem_usage=True)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

# Create a pipeline for text generation
try:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=-1
    )
    print("Pipeline created successfully.")
except Exception as e:
    print(f"Error creating pipeline: {e}")
    raise

# Create a LangChain LLM from the pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Load embeddings model
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Embeddings model loaded successfully.")
except Exception as e:
    print(f"Error loading embeddings model: {e}")
    raise

# Load your pre-built FAISS vector store
try:
    db = FAISS.load_local("vector_db", embeddings)
    print("FAISS vector store loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS vector store: {e}")
    raise

# Create a retriever for fetching relevant documents
retriever = db.as_retriever()

# Create a RetrievalQA chain
try:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("RetrievalQA chain created successfully.")
except Exception as e:
    print(f"Error creating RetrievalQA chain: {e}")
    raise

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.form.get('query')
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        result = qa.run(user_query)
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)

