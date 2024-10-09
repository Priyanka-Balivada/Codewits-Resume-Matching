# Import necessary libraries
import streamlit as st
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# Download necessary NLTK data
nltk.download('punkt')

# Define regular expressions for pattern matching
float_regex = re.compile(r'^\d{1,2}(\.\d{1,2})?$')
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
float_digit_regex = re.compile(r'^\d{10}$')
email_with_phone_regex = re.compile(r'(\d{10}).|.(\d{10})')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to tokenize text using the NLP model
def tokenize_text(text, nlp_model):
    doc = nlp_model(text, disable=["tagger", "parser"])
    tokens = [(token.text.lower(), token.label_) for token in doc.ents]
    return tokens

# Function to extract CGPA from a resume
def extract_cgpa(resume_text):
    cgpa_pattern = r'\b(?:CGPA|GPA|C\.G\.PA|Cumulative GPA)\s*:?[\s-]([0-9]+(?:\.[0-9]+)?)\b|\b([0-9]+(?:\.[0-9]+)?)\s(?:CGPA|GPA)\b'
    match = re.search(cgpa_pattern, resume_text, re.IGNORECASE)
    if match:
        cgpa = match.group(1) if match.group(1) else match.group(2)
        return float(cgpa)
    else:
        return None

# Function to extract skills from a resume
def extract_skills(text, skills_keywords):
    skills = [skill.lower() for skill in skills_keywords if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower())]
    return skills

# Function to preprocess text
def preprocess_text(text):
    return word_tokenize(text.lower())

# Function to train a Doc2Vec model
def train_doc2vec_model(documents):
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Function to calculate similarity between two texts
def calculate_similarity(model, text1, text2):
    vector1 = model.infer_vector(preprocess_text(text1))
    vector2 = model.infer_vector(preprocess_text(text2))
    return model.dv.cosine_similarities(vector1, [vector2])[0]

# Function to calculate accuracy
def accuracy_calculation(true_positives, false_positives, false_negatives):
    total = true_positives + false_positives + false_negatives
    accuracy = true_positives / total if total != 0 else 0
    return accuracy

# Streamlit Frontend
st.markdown("# Resume Matching Tool 📃📃")
st.markdown("An application to match resumes with a job description.")

# Sidebar - File Upload for Resumes
st.sidebar.markdown("## Upload Resumes PDF")
resumes_files = st.sidebar.file_uploader("Upload Resumes PDF", type=["pdf"], accept_multiple_files=True)

if resumes_files:
    # Sidebar - File Upload for Job Descriptions
    st.sidebar.markdown("## Upload Job Description PDF")
    job_descriptions_file = st.sidebar.file_uploader("Upload Job Description PDF", type=["pdf"])

    if job_descriptions_file:
        # Load the pre-trained NLP model
        nlp_model_path = "en_Resume_Matching_Keywords"
        nlp = spacy.load(nlp_model_path)

        # Backend Processing
        job_description_text = extract_text_from_pdf(job_descriptions_file)
        resumes_texts = [extract_text_from_pdf(resume_file) for resume_file in resumes_files]
        job_description_text = extract_text_from_pdf(job_descriptions_file)
        job_description_tokens = tokenize_text(job_description_text, nlp)

        # Initialize counters
        overall_skill_matches = 0
        overall_qualification_matches = 0

        # Create a list to store individual results
        results_list = []
        job_skills = set()
        job_qualifications = set()

        for job_token, job_label in job_description_tokens:
            if job_label == 'QUALIFICATION':
                job_qualifications.add(job_token.replace('\n', ' '))
            elif job_label == 'SKILLS':
                job_skills.add(job_token.replace('\n', ' '))

        job_skills_number = len(job_skills)
        job_qualifications_number = len(job_qualifications)

        # Lists to store counts of matched skills for all resumes
        skills_counts_all_resumes = []

        # Iterate over all uploaded resumes
        for uploaded_resume in resumes_files:
            resume_text = extract_text_from_pdf(uploaded_resume)
            resume_tokens = tokenize_text(resume_text, nlp)

            # Initialize counters for individual resume
            skillMatch = 0
            qualificationMatch = 0
            cgpa = ""

            # Lists to store matched skills and qualifications for each resume
            matched_skills = set()
            matched_qualifications = set()
            email = set()
            phone = set()
            name = set()

            # Compare the tokens in the resume with the job description
            for resume_token, resume_label in resume_tokens:
                for job_token, job_label in job_description_tokens:
                    if resume_token.lower().replace('\n', ' ') == job_token.lower().replace('\n', ' '):
                        if resume_label == 'SKILLS':
                            matched_skills.add(resume_token.replace('\n', ' '))
                        elif resume_label == 'QUALIFICATION':
                            matched_qualifications.add(resume_token.replace('\n', ' '))
                    elif resume_label == 'PHONE' and bool(float_digit_regex.match(resume_token)):
                        phone.add(resume_token)  
                    elif resume_label == 'QUALIFICATION':
                        matched_qualifications.add(resume_token.replace('\n', ' '))

            skillMatch = len(matched_skills)
            qualificationMatch = len(matched_qualifications)

            # Convert the list of emails to a set
            email_set = set(re.findall(email_pattern, resume_text.replace('\n', ' ')))
            email.update(email_set)

            numberphone=""
            for email_str in email:
                numberphone = email_with_phone_regex.search(email_str)
                if numberphone:
                    email.remove(email_str)
                    val=numberphone.group(1) or numberphone.group(2)
                    phone.add(val)
                    email.add(email_str.strip(val))

            # Increment overall counters based on matches
            overall_skill_matches += skillMatch
            overall_qualification_matches += qualificationMatch

            # Add count of matched skills for this resume to the list
            skills_counts_all_resumes.append([resume_text.count(skill.lower()) for skill in job_skills])

            # Create a dictionary for the current resume and append to the results list
            result_dict = {
                "Resume": uploaded_resume.name,
                "Similarity Score": (skillMatch/job_skills_number)*100,
                "Skill Matches": skillMatch,
                "Matched Skills": matched_skills,
                "CGPA": extract_cgpa(resume_text),
                "Email": email,
                "Phone": phone,
                "Qualification Matches": qualificationMatch,
                "Matched Qualifications": matched_qualifications
            }

            results_list.append(result_dict)

        # Display overall matches
        st.subheader("Overall Matches")
        st.write(f"Total Skill Matches: {overall_skill_matches}")
        st.write(f"Total Qualification Matches: {overall_qualification_matches}")
        st.write(f"Job Qualifications: {job_qualifications}")
        st.write(f"Job Skills: {job_skills}")

        # Display individual results in a table
        results_df = pd.DataFrame(results_list)
        st.subheader("Individual Results")
        st.dataframe(results_df)
        tagged_resumes = [TaggedDocument(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(resumes_texts)]
        model_resumes = train_doc2vec_model(tagged_resumes)

        st.subheader("\nHeatmap:")
       
        # Get skills keywords from user input
        skills_keywords_input = st.text_input("Enter skills keywords separated by commas (e.g., python, java, machine learning):")
        skills_keywords = [skill.strip() for skill in skills_keywords_input.split(',') if skill.strip()]

        if skills_keywords:
            # Calculate the similarity score between each skill keyword and the resume text
            skills_similarity_scores = []
            for resume_text in resumes_texts:
                resume_text_similarity_scores = []
                for skill in skills_keywords:
                    similarity_score = calculate_similarity(model_resumes, resume_text, skill)
                    resume_text_similarity_scores.append(similarity_score)
                skills_similarity_scores.append(resume_text_similarity_scores)

            # Create a DataFrame with the similarity scores and set the index to the names of the PDFs
            skills_similarity_df = pd.DataFrame(skills_similarity_scores, columns=skills_keywords, index=[resume_file.name for resume_file in resumes_files])

            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(skills_similarity_df, cmap='YlGnBu', annot=True, fmt=".2f", ax=ax)
            ax.set_title('Heatmap for Skills Similarity')
            ax.set_xlabel('Skills')
            ax.set_ylabel('Resumes')

            # Rotate the y-axis labels for better readability
            plt.yticks(rotation=0)

            # Display the Matplotlib figure using st.pyplot()
            st.pyplot(fig)
        else:
            st.write("Please enter at least one skill keyword.")

    else:
        st.warning("Please upload the Job Description PDF to proceed.")
else:
    st.warning("Please upload Resumes PDF to proceed.")
