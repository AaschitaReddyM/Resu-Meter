import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import time
import PyPDF2
import docx
from io import BytesIO

# --- AI Model and API Configuration ---
# Read the API key from the environment variable
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

@st.cache_resource
def load_model():
    """Loads the SentenceTransformer model once and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load the model
model = load_model()

# --- Helper Functions ---
def extract_text_from_file(uploaded_file):
    """Extracts text from txt, pdf, or docx files."""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            # Use BytesIO to handle the in-memory file
            pdf_file = BytesIO(uploaded_file.getvalue())
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(BytesIO(uploaded_file.getvalue()))
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\\n"
            return text
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None
    return None

def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_gemini_summary(job_description, resume_text):
    """Generates a concise summary using the Gemini API with exponential backoff."""
    prompt = (
        f"Based on the following job description and candidate resume, "
        f"write a concise summary (1-2 sentences) explaining why this candidate "
        f"is a great fit for the role.\\n\\n"
        f"Job Description:\\n{job_description}\\n\\n"
        f"Candidate Resume:\\n{resume_text}"
    )
    retries = 3
    delay = 2
    for i in range(retries):
        try:
            generation_config = genai.types.GenerationConfig(temperature=0.7)
            model_gemini = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
            response = model_gemini.generate_content(prompt)
            if response and response.text:
                return response.text
            else:
                return "Could not generate a summary from the API response."
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                return f"Error generating summary after multiple retries: {str(e)}"
    return "Error: Could not generate summary."


# --- Streamlit UI ---
st.set_page_config(page_title="Candidate Recommender", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:40px !important; font-weight: bold; color: #4A90E2; }
    .stButton>button { width: 100%; font-size: 1.25rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Resu-Meter : A Candidate Recommendation Engine</p>', unsafe_allow_html=True)
st.write("Hi Recruiter! Find the best candidates by uploading resumes or pasting text. The engine will rank them based on relevance to the job description.")

with st.form("recommendation_form"):
    st.subheader("1. Job Description")
    job_description = st.text_area("Please paste the job description here...", height=200, key="job_desc")

    st.subheader("2. Candidate Resumes")
    tab1, tab2 = st.tabs(["📄 Upload Resumes", "✍️ Paste Resumes"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Choose resume files (accepted formats : .pdf, .docx, .txt)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )

    with tab2:
        if 'candidates' not in st.session_state:
            st.session_state.candidates = [{'name': '', 'resume': ''}, {'name': '', 'resume': ''}]

        for i, candidate in enumerate(st.session_state.candidates):
            st.text_input(f"Candidate Name or Candidate ID", key=f"name_{i}", placeholder=f"Candidate {i+1} Name")
            st.text_area(f"Resume Text", height=150, key=f"resume_{i}", placeholder="Please paste resume text here...")
            st.markdown("---")

    submit_button = st.form_submit_button("✨ Generate Recommendations")

# --- Recommendation Logic ---
if submit_button:
    all_candidates = []

    # 1. Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_file(uploaded_file)
            if resume_text:
                all_candidates.append({'name': uploaded_file.name, 'resume': resume_text})

    # 2. Process manually pasted resumes
    pasted_candidates = [{'name': st.session_state[f'name_{i}'], 'resume': st.session_state[f'resume_{i}']} for i in range(len(st.session_state.candidates))]
    valid_pasted_candidates = [c for c in pasted_candidates if c['name'] and c['resume']]
    all_candidates.extend(valid_pasted_candidates)

    if not job_description or not all_candidates:
        st.error("❗ Please provide a job description and at least one resume (either uploaded or pasted).")
    else:
        st.subheader("🏆 Top Candidate Recommendations")
        with st.spinner('Analyzing resumes and generating recommendations... This may take a moment.'):
            job_embedding = model.encode(job_description)
            recommendations = []

            for candidate in all_candidates:
                resume_embedding = model.encode(candidate['resume'])
                similarity = cosine_similarity(job_embedding, resume_embedding)
                recommendations.append({
                    'name': candidate['name'],
                    'similarity': float(similarity),
                    'resume': candidate['resume']
                })

            recommendations.sort(key=lambda x: x['similarity'], reverse=True)
            top_recommendations = recommendations[:10] # Display top 10

        st.success("🎉 Recommendations generated!")
        for i, rec in enumerate(top_recommendations):
            st.markdown(f"### **{i+1}. {rec['name']}**")
            st.progress(rec['similarity'], text=f"**Relevance Score: {rec['similarity']:.2%}**")

            with st.expander("🤖 Show AI-Generated Summary"):
                with st.spinner('Generating summary...'):
                    summary = get_gemini_summary(job_description, rec['resume'])
                    st.write(summary)
