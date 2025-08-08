import pickle
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import joblib



# Custom CSS for fancy background and styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .big-text-area textarea {
        min-height: 200px !important;
        font-size: 1.2em !important;
    }
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Model selection dropdown at the top


st.markdown('<div class="centered">', unsafe_allow_html=True)
st.markdown(
    "<h1 style='color: white; text-shadow: 2px 2px 8px #333; white-space: nowrap;'>Review Sentiment Analyzer 🎬 /🛒</h1>",
    unsafe_allow_html=True
)


# Big, scrollable text area
review = st.text_area(
    "Enter your review below:",
    height=200,
    key="review_input",
    help="Type or paste your review here.",
    label_visibility="visible"
)
model_option = st.selectbox(
    "Choose a model to use for sentiment analysis:",
    ("Model 1: Hugging Face", "Model 2: Scikit Learn"),
    key="model_selector"
)

# Load models based on selection

st.markdown(
    """
    <style>
    textarea {
        min-height: 200px !important;
        font-size: 1.2em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Centered Analyze button
analyze = st.button("Analyze", key="analyze_btn")

st.markdown('</div>', unsafe_allow_html=True)
@st.cache_resource(show_spinner=False)
def load_sklearn_model():
    with open("sentiment_model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")




if analyze:
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    elif model_option == "Model 1: Hugging Face":
     sentiment_pipeline = load_hf_pipeline()  
    
     with st.spinner("Analyzing..."):
      result = sentiment_pipeline(review)[0]
     st.success(f"Sentiment: {result['label']},  Score: {result['score']:.2f}")


    elif model_option == "Model 2: Scikit Learn":
      Smodel, vectorizer = load_sklearn_model()
      review_vec = vectorizer.transform([review])
      result = Smodel.predict(review_vec)[0]
      proba = Smodel.predict_proba(review_vec)[0]
      st.success(f"Sentiment: {'positive' if result == 'pos' else 'negative'},   Score : {max(proba):.2f}")