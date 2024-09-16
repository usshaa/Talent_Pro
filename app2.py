from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with CPU only
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", low_cpu_mem_usage=True)

# Use the model for inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model(**inputs)
