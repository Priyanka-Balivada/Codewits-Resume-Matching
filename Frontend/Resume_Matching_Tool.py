import streamlit as st
import os
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re
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
    return max([float(match[0]) if match[0] else 0 for match in cgpa_matches], default=0)

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

def extract_skills(text):
    skills_keywords = ["python", "java", "machine learning",
                       "data analysis", "communication", "problem solving"]
    skills = [skill.lower()
              for skill in skills_keywords if skill.lower() in text.lower()]
    return skills

# Streamlit Frontend
st.title("Resume Matching Tool")

# Sidebar - Select Job Descriptions Folder
job_descriptions_folder = st.sidebar.text_input("Enter the path or name of the job descriptions folder")

# Sidebar - Select Resumes Folder
resumes_folder = st.sidebar.text_input("Enter the path or name of the resumes folder")

# Sidebar - Select Excel Output Folder
output_folder = st.sidebar.text_input("Enter the path or name of the output folder for Excel file", "./")

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
    cgpa = extract_cgpa(resume_text)
    results_data['Resume'].append(os.path.basename(resumes_files[i]))
    smScore = similarity_score * 100
    total_score = smScore + cgpa
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

# Calculate skills for all resumes
all_resumes_skills = [extract_skills(resume_text)
                      for resume_text in resumes_texts]

# Create a DataFrame for skills distribution
skills_distribution_data = {'Resume': [], 'Skill': [], 'Frequency': []}
for i, resume_skills in enumerate(all_resumes_skills):
    for skill in set(resume_skills):
        skills_distribution_data['Resume'].append(
            os.path.basename(resumes_files[i]))
        skills_distribution_data['Skill'].append(skill)
        skills_distribution_data['Frequency'].append(
            resume_skills.count(skill))

skills_distribution_df = pd.DataFrame(skills_distribution_data)

# Pivot the DataFrame for heatmap
skills_heatmap_df = skills_distribution_df.pivot(
    index='Resume', columns='Skill', values='Frequency').fillna(0)

# Normalize the values for better visualization
skills_heatmap_df_normalized = skills_heatmap_df.div(
    skills_heatmap_df.sum(axis=1), axis=0)

# Plot the heatmap
st.subheader("Heatmap for Skills Distribution", divider="rainbow")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(skills_heatmap_df_normalized,
            cmap='YlGnBu', annot=True, fmt=".2f", ax=ax)
ax.set_xlabel('Resume')
ax.set_ylabel('Skill')

# Display the Matplotlib figure using st.pyplot()
st.pyplot(fig)
