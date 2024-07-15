import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import string
import heapq
import spacy
from io import BytesIO
import fitz
from docx import Document
from collections import Counter
import re
 
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")
 
 
# Function to extract text from PDF
def extract_text_from_pdf(upload_file):
    try:
        pdf_data = BytesIO(upload_file.read())
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""
 
# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    doc_text = []
    for paragraph in doc.paragraphs:
        doc_text.append(paragraph.text)
    return '\n'.join(doc_text)
 
# Function to preprocess text
def preprocess_text(text):
    # Tokenize into words
    words = word_tokenize(text)
    # Convert to lowercase
    words = [word.lower() for word in words]
    # Remove punctuation
    words = [word for word in words if word not in string.punctuation]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)
 
 
# Function to generate summary 
def generate_summary(text, num_sentences=7):
    sentences = sent_tokenize(text)
    preprocessed_text = preprocess_text(text)
    word_freq = Counter(preprocessed_text.split())
    total_words = sum(word_freq.values())

    # Calculate TF-IDF scores
    word_tfidf = {word: freq/total_words * (1 + sentences.count(word)) for word, freq in word_freq.items()}
    sent_scores = {}
    for sentence in sentences:
        for word in preprocess_text(sentence).split():
            if word in word_tfidf:
                if len(sentence.split()) < 30:
                    sent_scores[sentence] = sent_scores.get(sentence, 0) + word_tfidf[word]

                    
    # Getting the highest scores
    summary_sentences = heapq.nlargest(num_sentences, sent_scores, key=sent_scores.get)
    summary_sentences = sorted(summary_sentences, key=lambda x: sentences.index(x))

    rephrased_sentences = []
    for sentence in summary_sentences:
        doc = nlp(sentence)
        # Extracting the subject or verb
        subject = ""
        verb = ""
        for token in doc:
            if token.dep_ == "nsubj":
                subject = token.text
            if token.pos_ == "VERB":
                verb = token.text
                break

        # Rephrasing
        if subject and verb:
            rephrased = f"Regarding {subject}, it {verb}"
            remaining_words = [token.text for token in doc if token.text not in [subject, verb] and token.pos_ not in ["DET", "ADP", "CONJ", "CCONJ"]]
            rephrased += " " + " ".join(remaining_words)
        else:
            rephrased = sentence
        rephrased_sentences.append(rephrased)
    summary = ' '.join(rephrased_sentences)
    return summary
 

# Streamlit app
def main():
    st.title("Document Summarization App")
 
    # Custom CSS for button color
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #006400;  /* Dark green */
            color: white;
            font-color: yellow;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
 
    # File upload and text area for user
    uploaded_file = st.file_uploader("Upload Files (PDF, DOCX)", type=['pdf', 'docx'])
 
    text=""
    text = st.text_area("Or Enter Text Here:")
   
   
   
    # Extract text from uploaded file
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        if file_type == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            text = extract_text_from_docx(uploaded_file)
       
   
   
    # Summarize button
    if st.button("Summarize"):
        if not text.strip():  # Check if text is empty or just whitespace
            st.write("Oops! Please upload a file or enter text.")
        else:
            summarized_text = generate_summary(text)
            st.subheader("Summarized Text:")
            st.write(summarized_text)
       
       
if __name__ == '__main__':
    main()