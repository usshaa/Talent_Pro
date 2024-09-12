from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfFolder
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

# Authenticate with Hugging Face
HfFolder.save_token(hf_token)  # Save token for later use

# Define summarizer using transformers with explicit model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the Mistral model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def summarize_text(text_logs):
    summarized = summarizer(text_logs, max_length=50, min_length=25, do_sample=False)
    return summarized[0]['summary_text']

# Load learner log data (CSV)
def load_logs():
    df = pd.read_csv('logs/learner_logs.csv')
    return df

# Analyze sentiment using model
def analyze_learner_sentiment(text_logs):
    # Summarize or reduce the size of text_logs if necessary
    shortened_logs = summarize_text(text_logs)
    
    # Prepare the prompt for the model
    prompt = f"Analyze the following text data to extract learner sentiment and engagement:\n\n{shortened_logs}"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the response using the model
    outputs = model.generate(**inputs, max_length=100)
    
    # Decode the output to text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response_text

# Generate learner performance graph
def plot_performance_trend(data):
    fig = px.line(data, x='date', y='score', title='Learner Performance Over Time')
    fig_html = pio.to_html(fig, full_html=False)
    return fig_html

# Flask route for dashboard
@app.route('/')
def dashboard():
    logs_df = load_logs()
    
    # Preprocess logs for trend analysis (simplified)
    performance_trend = logs_df[['date', 'score']]
    
    # Plot the performance trend
    performance_graph = plot_performance_trend(performance_trend)
    
    # Generate sentiment analysis based on interactions (simplified)
    interaction_logs = logs_df['interactions'].dropna().str.cat(sep=' ')
    sentiment_analysis = analyze_learner_sentiment(interaction_logs)

    return render_template('dashboard.html', 
                           performance_graph=performance_graph, 
                           sentiment_analysis=sentiment_analysis)

if __name__ == '__main__':
    app.run(debug=True)
