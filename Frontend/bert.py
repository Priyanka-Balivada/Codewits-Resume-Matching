import os
import nltk
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from transformers import BertTokenizer, BertModel
import torch

nltk.download('punkt')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def preprocess_text(text):
    return word_tokenize(text.lower())

def extract_cgpa(text):
    cgpa_pattern = r'\b\d\.\d{1,2}\b'  # Pattern for CGPA (e.g., 3.75, 4.0, etc.)
    cgpa_matches = re.findall(cgpa_pattern, text)
    return cgpa_matches

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'

def calculate_similarity_bert(text1, text2):
    encoded_input = tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encoded_input)
    embeddings = outputs.last_hidden_state
    similarity_score = cosine_similarity(embeddings[0][0].unsqueeze(0), embeddings[0][1].unsqueeze(0)).item()
    return similarity_score

# Title of the application
st.title('Resume Matching Tool')

# Function to create vertical space
def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')

v_spacer(height=3, sb=False)
st.subheader("Choose Files")
uploaded_file = st.file_uploader("Choose a Job Description")

# Create a button to browse for a folder
v_spacer(height=3, sb=False)
clicked = st.button('Browse Resume Folder')

# Initialize the folder variable
selected_folder = ""

# Check if the button is clicked
if clicked:
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    # Ask the user to select a folder
    selected_folder = filedialog.askdirectory(master=root)

    # Display the selected folder
    st.write(f'Selected Folder: {selected_folder}')

# Check if both the job description and resume folder are selected
if uploaded_file and selected_folder:
    # Process job description
    try:
        job_description_text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            job_description_text = uploaded_file.read().decode("latin-1")
        except UnicodeDecodeError:
            job_description_text = uploaded_file.read().decode("utf-16")

    # Process resumes
    resumes_files = [os.path.join(selected_folder, file) for file in os.listdir(selected_folder) if file.endswith(".pdf")]

    # Create a DataFrame to store the results
    results_data = {'Resume': [], 'Similarity Score': [], 'CGPA': [], 'Total Score': [], 'Email': [], 'Contact': []}

    # Compare the selected job description with all resumes
    for i, resume_file in enumerate(resumes_files):
        resume_text = extract_text_from_pdf(resume_file)
        similarity_score = calculate_similarity_bert(job_description_text, resume_text)
        cgpa_matches = extract_cgpa(resume_text)
        cgpa = ', '.join(cgpa_matches) if cgpa_matches else 'Not found'
        results_data['Resume'].append(os.path.basename(resume_file))
        results_data['Similarity Score'].append(similarity_score * 100)
        results_data['CGPA'].append(cgpa)

        emails = re.findall(email_pattern, resume_text)
        contacts = re.findall(phone_pattern, resume_text)
        if cgpa == "Not found":
            results_data['Total Score'].append(similarity_score * 100)
        else:
            total_score = float(cgpa) + similarity_score * 100
            results_data['Total Score'].append(total_score)

        results_data['Email'].append(emails)
        results_data['Contact'].append(contacts)

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results_data)

    # Sort the DataFrame by similarity score in descending order
    results_df = results_df.sort_values(by='Similarity Score', ascending=False)

    # Display the results table
    st.table(results_df)

    # Download results
    if st.button('Download Results'):
        results_df.to_excel('ResumeMatched.xlsx', index=False)

    v_spacer(height=3, sb=False)

    # Plot a Bar Chart to show the distribution of similarity scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df['Resume'], results_df['Similarity Score'], color='blue')
    ax.set_xlabel('Resume')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Similarity Scores between Job Description and Resumes')

    # Set the tick locations and labels
    ax.set_xticks(range(len(results_df['Resume'])))
    ax.set_xticklabels(results_df['Resume'], rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig)

    # Plot a Pie Chart to show the distribution of similarity scores
    plt.figure(figsize=(8, 8))

    # Normalize similarity scores to be non-negative
    normalized_scores = results_df['Similarity Score'] - results_df['Similarity Score'].min()
    total_similarity = normalized_scores.sum()

    # Check if all similarity scores are zero
    if total_similarity == 0:
        # If all scores are zero, assign equal weights
        normalized_scores = 1
    else:
        # Normalize scores
        normalized_scores /= total_similarity

    plt.pie(normalized_scores, labels=results_df['Resume'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
    plt.title('Distribution of Similarity Scores with Job Description')
    st.pyplot(plt)

    # Create a Knowledge Graph using NetworkX
    G = nx.Graph()

    # Add nodes for job description and resumes
    G.add_node('Job Description', color='blue', size=100)
    for i, resume_path in enumerate(resumes_files):
        G.add_node(os.path.basename(resume_path), color='green', size=30)

    # Add edges with similarity scores as weights
    for i, resume_file in enumerate(resumes_files):
        resume_text = extract_text_from_pdf(resume_file)
        similarity_score = calculate_similarity_bert(job_description_text, resume_text)
        G.add_edge('Job Description', os.path.basename(resumes_files[i]), weight=similarity_score)

    # Plot the Knowledge Graph
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, font_size=8, node_size=[d['size'] for n, d in G.nodes(data=True)], node_color=[d['color'] for n, d in G.nodes(data=True)], ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
    ax.set_title('Knowledge Graph: Similarity Relationships between Job Description and Resumes')
    st.pyplot(fig)
