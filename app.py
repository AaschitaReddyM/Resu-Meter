import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import time

# --- AI Model and API Configuration ---
# Read the API key from the environment variable set by the launcher script
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

@st.cache_resource
def load_model():
    """Loads the SentenceTransformer model once and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load the model
model = load_model()

# --- Core Functions ---
def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def get_gemini_summary(job_description, resume_text):
    """
    Generates a concise summary using the Gemini API.
    Implements exponential backoff for API calls.
    """
    prompt = (
        f"Based on the following job description and candidate resume, "
        f"write a concise summary (1-2 sentences) explaining why this candidate "
        f"is a great fit for the role.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate Resume:\n{resume_text}"
    )

    retries = 0
    while retries < 5:
        try:
            response = genai.GenerativeModel('gemini-2.5-flash-preview-05-20').generate_content(prompt)
            # Check if the response is valid and return the text
            if response and response.candidates and len(response.candidates) > 0:
                return response.text
            else:
                return "Could not generate summary."
        except Exception as e:
            retries += 1
            delay = 2 ** retries  # Exponential backoff
            time.sleep(delay)
            if retries == 5:
                return f"Error generating summary: {str(e)}"
    return "Error generating summary after multiple retries."

# --- Streamlit UI ---
st.set_page_config(
    page_title="Candidate Recommender",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
        font-weight: bold;
        color: #4A90E2;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.25rem;
        font-weight: bold;
        color: white;
        background-color: #4A90E2;
        border-radius: 0.75rem;
    }
    .st-emotion-cache-1wv7c0w {
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Candidate Recommendation Engine</p>', unsafe_allow_html=True)
st.write("Find the best candidates for your job opening using AI-powered semantic search.")

# --- Form for inputs ---
with st.form("recommendation_form"):
    st.subheader("Job Description")
    job_description = st.text_area("Paste the job description here...", height=250, key="job_desc")

    st.subheader("Candidate Resumes")

    # Store candidates in session state
    if 'candidates' not in st.session_state:
        st.session_state.candidates = [{'name': '', 'resume': ''}, {'name': '', 'resume': ''}]

    for i, candidate in enumerate(st.session_state.candidates):
        st.markdown("---")
        with st.container():
            st.text_input(f"Candidate Name or ID", key=f"name_{i}", value=candidate['name'])
            st.text_area(f"Resume Text", height=200, key=f"resume_{i}", value=candidate['resume'])

    submit_button = st.form_submit_button("Generate Recommendations")

# --- Recommendation Logic ---
if submit_button:
    # Now we retrieve the latest values from the session state keys
    st.session_state.candidates = [{'name': st.session_state[f'name_{i}'], 'resume': st.session_state[f'resume_{i}']} for i in range(len(st.session_state.candidates))]
    valid_candidates = [c for c in st.session_state.candidates if c['name'] and c['resume']]

    if not job_description or not valid_candidates:
        st.error("Please provide a job description and at least one complete candidate entry.")
    else:
        st.subheader("Top Candidate Recommendations")
        with st.spinner('Generating recommendations...'):
            # Generate embedding for the job description
            job_embedding = model.encode(job_description)

            recommendations = []
            for candidate in valid_candidates:
                resume_embedding = model.encode(candidate['resume'])
                similarity = cosine_similarity(job_embedding, resume_embedding)
                recommendations.append({
                    'name': candidate['name'],
                    'similarity': float(similarity),
                    'resume': candidate['resume']
                })

            # Sort and get top 10
            recommendations.sort(key=lambda x: x['similarity'], reverse=True)
            top_recommendations = recommendations[:10]

        # Display results in a table
        st.success("Recommendations generated!")
        for i, rec in enumerate(top_recommendations):
            st.markdown(f"### {i+1}. {rec['name']} - Score: {rec['similarity']:.4f}")
            with st.expander("Show AI Summary"):
                with st.spinner('Generating summary...'):
                    summary = get_gemini_summary(job_description, rec['resume'])
                    st.write(summary)
