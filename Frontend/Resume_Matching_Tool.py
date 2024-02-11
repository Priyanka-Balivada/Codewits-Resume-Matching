import streamlit as st
import os
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re  # Added import for regular expressions
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def preprocess_text(text):
    return word_tokenize(text.lower())

# Function to extract CGPA from text using regular expressions
def extract_cgpa(text):
    cgpa_pattern = r'\b(\d\.\d{1,2})/\d{1,2}|\b(\d\.\d{1,2})\b'  # Pattern for CGPA (e.g., 3.75, 4.0, etc.)
    cgpa_matches = re.findall(cgpa_pattern, text)
    return [float(match[0]) if match[0] else 0 for match in cgpa_matches]

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'

def train_doc2vec_model(documents):
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def calculate_similarity(model, text1, text2):
    vector1 = model.infer_vector(preprocess_text(text1))
    vector2 = model.infer_vector(preprocess_text(text2))
    return model.dv.cosine_similarities(vector1, [vector2])[0]

def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')

# Streamlit Frontend
st.title("Resume Matching Tool")

# Sidebar - Select Job Descriptions Folder
job_descriptions_folder = st.sidebar.text_input("Enter the path or name of the job descriptions folder")

# Sidebar - Select Resumes Folder
resumes_folder = st.sidebar.text_input("Enter the path or name of the resumes folder")

# Sidebar - Select Excel Output Folder
output_folder = st.sidebar.text_input("Enter the path or name of the output folder for Excel file","./")

# Sidebar - Sorting Options
sort_options = ['Similarity Score', 'CGPA', 'Total Score']
selected_sort_option = st.sidebar.selectbox("Sort results by", sort_options)

job_descriptions_files = [os.path.join(job_descriptions_folder, file) for file in os.listdir(job_descriptions_folder) if file.endswith(".pdf")]
selected_job_file = st.sidebar.selectbox("Choose a job description", job_descriptions_files, format_func=lambda x: os.path.basename(x))
selected_job_text = extract_text_from_pdf(selected_job_file)

# Backend Processing
resumes_files = [os.path.join(resumes_folder, file) for file in os.listdir(resumes_folder) if file.endswith(".pdf")]
resumes_texts = [extract_text_from_pdf(resume_path) for resume_path in resumes_files]

tagged_resumes = [TaggedDocument(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(resumes_texts)]
model_resumes = train_doc2vec_model(tagged_resumes)

results_data = {'Resume': [], 'Similarity Score': [], 'CGPA': [], 'Total Score': [], 'Email': [], 'Contact': []}

for i, resume_text in enumerate(resumes_texts):
    similarity_score = calculate_similarity(model_resumes, resume_text, selected_job_text)
    cgpa_values = extract_cgpa(resume_text)
    cgpa = ', '.join(map(str, cgpa_values)) if cgpa_values else '0'
    results_data['Resume'].append(os.path.basename(resumes_files[i]))
    smScore = similarity_score * 100
    total_score = smScore + sum(cgpa_values)
    results_data['Similarity Score'].append(smScore)
    results_data['CGPA'].append(cgpa)
    results_data['Total Score'].append(total_score)

    emails = ', '.join(re.findall(email_pattern, resume_text))
    contacts = ', '.join(re.findall(phone_pattern, resume_text))
    results_data['Email'].append(emails)
    results_data['Contact'].append(contacts)

# Create a DataFrame
results_df = pd.DataFrame(results_data)

# Sort the DataFrame based on user-selected option
if selected_sort_option == 'Similarity Score':
    results_df = results_df.sort_values(by='Similarity Score', ascending=False)
elif selected_sort_option == 'CGPA':
    results_df = results_df.sort_values(by='CGPA', ascending=False)
else:
    results_df = results_df.sort_values(by='Total Score', ascending=False)

# Display the results table with job description name
st.subheader(f"Results Table for Job: {os.path.basename(selected_job_file)} (sorted by {selected_sort_option} in descending order):")
st.table(results_df)

# Download Button for Excel with job description name
excel_file_name = f"results_{os.path.basename(selected_job_file).replace('.pdf', '')}.xlsx"
if st.button(f"Download Results as Excel ({excel_file_name})"):
    output_path = os.path.join(output_folder, excel_file_name)
    results_df.to_excel(output_path, index=False)
    st.success(f"Results downloaded successfully! File saved at {output_path}")

v_spacer(height=5, sb=False)

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

v_spacer(height=5, sb=False)

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


v_spacer(height=5, sb=False)
# Create a Knowledge Graph using NetworkX
G = nx.Graph()

# Add nodes for job description and resumes
G.add_node('Job Description', color='blue', size=100)
for i, resume_path in enumerate(resumes_files):
    G.add_node(os.path.basename(resume_path), color='green', size=30)

# Add edges with similarity scores as weights
for i, resume_file in enumerate(resumes_files):
    similarity_score = calculate_similarity(model_resumes, extract_text_from_pdf(resume_file), selected_job_text)
    G.add_edge('Job Description', os.path.basename(resumes_files[i]), weight=similarity_score)

# Plot the Knowledge Graph
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, font_size=8, node_size=[d['size'] for n, d in G.nodes(data=True)], node_color=[d['color'] for n, d in G.nodes(data=True)], ax=ax)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
ax.set_title('Knowledge Graph: Similarity Relationships between Job Description and Resumes')
st.pyplot(fig)
