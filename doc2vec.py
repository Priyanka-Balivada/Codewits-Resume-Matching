# Importing necessary libraries
from collections import Counter
import streamlit as st
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Downloading the 'punkt' tokenizer from NLTK
nltk.download('punkt')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to extract skills from a text using a list of skill keywords
def extract_skills(text, skills_keywords):
    skills = [skill.lower()
              for skill in skills_keywords if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower())]
    return skills

# Function to preprocess text by tokenizing and converting to lowercase
def preprocess_text(text):
    return word_tokenize(text.lower())

# Function to extract mobile numbers from a text
def extract_mobile_numbers(text):
    mobile_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    return re.findall(mobile_pattern, text)

# Function to extract emails from a text
def extract_emails(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

# Function to train a Doc2Vec model on a list of tagged documents
def train_doc2vec_model(documents):
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model

# Function to calculate the cosine similarity between two texts using a trained Doc2Vec model
def calculate_similarity(model, text1, text2):
    vector1 = model.infer_vector(preprocess_text(text1))
    vector2 = model.infer_vector(preprocess_text(text2))
    return model.dv.cosine_similarities(vector1, [vector2])[0]

# Function to calculate accuracy based on true positives, false positives, and false negatives
def accuracy_calculation(true_positives, false_positives, false_negatives):
    total = true_positives + false_positives + false_negatives
    accuracy = true_positives / total if total != 0 else 0
    return accuracy

# Function to extract CGPA from a text
def extract_cgpa(resume_text):
    # Define a regular expression pattern for CGPA extraction
    cgpa_pattern = r'\b(?:CGPA|GPA|C.G.PA|Cumulative GPA)\s*:?[\s-]* ([0-9]+(?:\.[0-9]+)?)\b|\b([0-9]+(?:\.[0-9]+)?)\s*(?:CGPA|GPA)\b'

    # Search for CGPA pattern in the text
    match = re.search(cgpa_pattern, resume_text, re.IGNORECASE)

    # Check if a match is found
    if match:
        cgpa = match.group(1)
        if cgpa is not None:
            return float(cgpa)
        else:
            return float(match.group(2))
    else:
        return None

# Regular expressions for email and phone number patterns
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'

# Streamlit Frontend
st.markdown("# Resume Matching Tool 📃📃")
st.markdown("An application to match resumes with a job description.")

# Sidebar - File Upload for Resumes
st.sidebar.markdown("## Upload Resumes PDF")
resumes_files = st.sidebar.file_uploader(
    "Upload Resumes PDF", type=["pdf"], accept_multiple_files=True)

if resumes_files:
    # Sidebar - File Upload for Job Descriptions
    st.sidebar.markdown("## Upload Job Description PDF")
    job_descriptions_file = st.sidebar.file_uploader(
        "Upload Job Description PDF", type=["pdf"])

    if job_descriptions_file:
        # Sidebar - Sorting Options
        sort_options = ['Weighted Score', 'Similarity Score']
        selected_sort_option = st.sidebar.selectbox(
            "Sort results by", sort_options)

        # Backend Processing
        job_description_text = extract_text_from_pdf(job_descriptions_file)
        resumes_texts = [extract_text_from_pdf(
            resume_file) for resume_file in resumes_files]

        tagged_resumes = [TaggedDocument(words=preprocess_text(
            text), tags=[str(i)]) for i, text in enumerate(resumes_texts)]
        model_resumes = train_doc2vec_model(tagged_resumes)

        true_positives_mobile = 0
        false_positives_mobile = 0
        false_negatives_mobile = 0

        true_positives_email = 0
        false_positives_email = 0
        false_negatives_email = 0

        results_data = {'Resume': [], 'Similarity Score': [],
                        'Weighted Score': [], 'Email': [], 'Contact': [], 'CGPA': []}

        for i, resume_text in enumerate(resumes_texts):
            extracted_mobile_numbers = set(extract_mobile_numbers(resume_text))
            extracted_emails = set(extract_emails(resume_text))
            extracted_cgpa = extract_cgpa(resume_text)

            ground_truth_mobile_numbers = {'1234567890', '9876543210'}
            ground_truth_emails = {
                'john.doe@example.com', 'jane.smith@example.com'}

            true_positives_mobile += len(
                extracted_mobile_numbers.intersection(ground_truth_mobile_numbers))
            false_positives_mobile += len(
                extracted_mobile_numbers.difference(ground_truth_mobile_numbers))
            false_negatives_mobile += len(
                ground_truth_mobile_numbers.difference(extracted_mobile_numbers))

            true_positives_email += len(
                extracted_emails.intersection(ground_truth_emails))
            false_positives_email += len(
                extracted_emails.difference(ground_truth_emails))
            false_negatives_email += len(
                ground_truth_emails.difference(extracted_emails))

            similarity_score = calculate_similarity(
                model_resumes, resume_text, job_description_text)

            other_criteria_score = 0

            weighted_score = (0.6 * similarity_score) + \
                (0.4 * other_criteria_score)

            results_data['Resume'].append(resumes_files[i].name)
            results_data['Similarity Score'].append(similarity_score * 100)
            results_data['Weighted Score'].append(weighted_score)

            emails = ', '.join(re.findall(email_pattern, resume_text))
            contacts = ', '.join(re.findall(phone_pattern, resume_text))
            results_data['Email'].append(emails)
            results_data['Contact'].append(contacts)
            results_data['CGPA'].append(extracted_cgpa)

        results_df = pd.DataFrame(results_data)

        if selected_sort_option == 'Similarity Score':
            results_df = results_df.sort_values(
                by='Similarity Score', ascending=False)
        else:
            results_df = results_df.sort_values(
                by='Weighted Score', ascending=False)

        st.subheader(f"Results Table (Sorted by {selected_sort_option}):")

        # Define a custom function to highlight maximum values in the specified columns
        def highlight_max(data, color='grey'):
            is_max = data == data.max()
            return [f'background-color: {color}' if val else '' for val in is_max]

        # Apply the custom highlighting function to the DataFrame
        st.dataframe(results_df.style.apply(highlight_max, subset=[
                     'Similarity Score', 'Weighted Score', 'CGPA']))


        highest_score_index = results_df['Similarity Score'].idxmax()
        highest_score_resume_name = resumes_files[highest_score_index].name

        st.subheader("\nDetails of Highest Similarity Score Resume:")
        st.write(f"Resume Name: {highest_score_resume_name}")
        st.write(
            f"Similarity Score: {results_df.loc[highest_score_index, 'Similarity Score']:.2f}")

        if 'Weighted Score' in results_df.columns:
            weighted_score_value = results_df.loc[highest_score_index,
                                                  'Weighted Score']
            st.write(f"Weighted Score: {weighted_score_value:.2f}" if pd.notnull(
                weighted_score_value) else "Weighted Score: Not Mentioned")
        else:
            st.write("Weighted Score: Not Mentioned")

        if 'Email' in results_df.columns:
            email_value = results_df.loc[highest_score_index, 'Email']
            st.write(f"Email: {email_value}" if pd.notnull(
                email_value) else "Email: Not Mentioned")
        else:
            st.write("Email: Not Mentioned")

        if 'Contact' in results_df.columns:
            contact_value = results_df.loc[highest_score_index, 'Contact']
            st.write(f"Contact: {contact_value}" if pd.notnull(
                contact_value) else "Contact: Not Mentioned")
        else:
            st.write("Contact: Not Mentioned")

        if 'CGPA' in results_df.columns:
            cgpa_value = results_df.loc[highest_score_index, 'CGPA']
            st.write(f"CGPA: {cgpa_value}" if pd.notnull(
                cgpa_value) else "CGPA: Not Mentioned")
        else:
            st.write("CGPA: Not Mentioned")

        mobile_accuracy = accuracy_calculation(
            true_positives_mobile, false_positives_mobile, false_negatives_mobile)
        email_accuracy = accuracy_calculation(
            true_positives_email, false_positives_email, false_negatives_email)

        st.subheader("\nHeatmap:")
        # st.write(f"Mobile Number Accuracy: {mobile_accuracy:.2%}")
        # st.write(f"Email Accuracy: {email_accuracy:.2%}")

        # Get skills keywords from user input
        skills_keywords_input = st.text_input(
            "Enter skills keywords separated by commas (e.g., python, java, machine learning):")
        skills_keywords = [skill.strip()
                        for skill in skills_keywords_input.split(',') if skill.strip()]

        if skills_keywords:
            # Calculate the similarity score between each skill keyword and the resume text
            skills_similarity_scores = []
            for resume_text in resumes_texts:
                resume_text_similarity_scores = []
                for skill in skills_keywords:
                    similarity_score = calculate_similarity(
                        model_resumes, resume_text, skill)
                    resume_text_similarity_scores.append(similarity_score)
                skills_similarity_scores.append(resume_text_similarity_scores)

            # Create a DataFrame with the similarity scores and set the index to the names of the PDFs
            skills_similarity_df = pd.DataFrame(
                skills_similarity_scores, columns=skills_keywords, index=[resume_file.name for resume_file in resumes_files])

            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(12, 8))

            sns.heatmap(skills_similarity_df,
                        cmap='YlGnBu', annot=True, fmt=".2f", ax=ax)
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
