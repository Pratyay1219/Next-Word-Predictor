import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Quote Generator · AI",
    page_icon="✦",
    layout="centered"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
  }

  /* ── Background ── */
  .stApp {
    background-color: #0b0b0f;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(180,140,80,0.12) 0%, transparent 70%),
      radial-gradient(ellipse 40% 30% at 80% 90%, rgba(120,80,200,0.06) 0%, transparent 60%);
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 720px; }

  /* ── Header section ── */
  .hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
  }
  .hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #b8922a;
    margin-bottom: 1.2rem;
    display: block;
  }
  .hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 6vw, 4.4rem);
    font-weight: 300;
    color: #f0e8d5;
    line-height: 1.1;
    margin: 0 0 0.6rem;
    letter-spacing: -0.01em;
  }
  .hero-title em {
    font-style: italic;
    color: #c9a84c;
  }
  .hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6b6570;
    letter-spacing: 0.08em;
    margin-top: 1rem;
  }
  .divider {
    width: 40px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #b8922a, transparent);
    margin: 1.8rem auto;
  }

  /* ── Input card ── */
  .input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(180,140,80,0.15);
    border-radius: 4px;
    padding: 2rem 2.4rem 2.2rem;
    position: relative;
    overflow: hidden;
  }
  .input-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(180,140,80,0.5), transparent);
  }
  .input-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #b8922a;
    margin-bottom: 0.8rem;
    display: block;
  }

  /* ── Streamlit text_input override ── */
  .stTextInput > div > div > input {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(180,140,80,0.22) !important;
    border-radius: 3px !important;
    color: #f0e8d5 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.15rem !important;
    font-style: italic !important;
    padding: 0.85rem 1.1rem !important;
    letter-spacing: 0.01em !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: rgba(180,140,80,0.55) !important;
    box-shadow: 0 0 0 3px rgba(180,140,80,0.06) !important;
    outline: none !important;
  }
  .stTextInput > div > div > input::placeholder {
    color: #3d3840 !important;
    font-style: italic !important;
  }
  .stTextInput label { display: none !important; }

  /* ── Button override ── */
  .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(180,140,80,0.45) !important;
    color: #c9a84c !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 2px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    margin-top: 1rem !important;
  }
  .stButton > button:hover {
    background: rgba(180,140,80,0.08) !important;
    border-color: rgba(180,140,80,0.75) !important;
    color: #e8c97a !important;
    box-shadow: 0 0 20px rgba(180,140,80,0.1) !important;
  }
  .stButton > button:active {
    background: rgba(180,140,80,0.15) !important;
    transform: scale(0.99) !important;
  }

  /* ── Output quote ── */
  .quote-container {
    margin: 2rem 0;
    padding: 2.5rem 2.8rem;
    background: rgba(180,140,80,0.04);
    border-left: 2px solid #b8922a;
    border-radius: 0 4px 4px 0;
    position: relative;
    animation: fadeUp 0.6s ease forwards;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .quote-mark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem;
    line-height: 0;
    color: rgba(180,140,80,0.2);
    position: absolute;
    top: 1.8rem;
    left: 1.6rem;
    font-style: italic;
  }
  .quote-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-style: italic;
    font-weight: 300;
    color: #f0e8d5;
    line-height: 1.6;
    padding-left: 1.2rem;
    letter-spacing: 0.01em;
  }
  .quote-badge {
    display: inline-block;
    margin-top: 1.2rem;
    margin-left: 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #b8922a;
    opacity: 0.7;
  }

  /* ── Warning / info override ── */
  .stAlert {
    background: rgba(180,140,80,0.05) !important;
    border: 1px solid rgba(180,140,80,0.2) !important;
    border-radius: 3px !important;
    color: #c9a84c !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
  }

  /* ── Footer ── */
  .footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
  }
  .footer-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #3d3840;
    text-transform: uppercase;
  }
  .footer-dot {
    color: #b8922a;
    margin: 0 0.5rem;
  }

  /* ── Spinner override ── */
  .stSpinner > div {
    border-top-color: #b8922a !important;
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------
# Load resources
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model (1).h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()


# ------------------------------
# Generate full sentence
# ------------------------------
def generate_full_sentence(seed_text):
    generated_text = seed_text
    while True:
        sequence = tokenizer.texts_to_sequences([generated_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
        preds = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(preds)
        output_word = tokenizer.index_word.get(predicted_index, "")
        if output_word == "":
            break
        generated_text = generated_text + " " + output_word
        if output_word in [".", "!", "?"]:
            break
        if len(generated_text.split()) > 30:
            break
    return generated_text


# ------------------------------
# UI Layout
# ------------------------------

# Hero
st.markdown("""
<div class="hero">
  <span class="hero-eyebrow">✦ Neural Text Intelligence ✦</span>
  <h1 class="hero-title">Quote <em>Generator</em></h1>
  <p class="hero-sub">LSTM · Trained on curated quotes · Greedy decoding</p>
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# Input card
# st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<span class="input-label">✦ Seed your thought</span>', unsafe_allow_html=True)

user_input = st.text_input(
    label="seed",
    placeholder="e.g.  the only way to live is...",
    label_visibility="collapsed"
)

generate_clicked = st.button("✦  Generate Quote  ✦")
st.markdown('</div>', unsafe_allow_html=True)

# Output
if generate_clicked:
    if not user_input.strip():
        st.warning("⚠  Please enter a seed phrase to begin.")
    else:
        with st.spinner("Composing..."):
            result = generate_full_sentence(user_input)

        st.markdown(f"""
        <div class="quote-container">
          <span class="quote-mark">"</span>
          <p class="quote-text">{result}</p>
          <span class="quote-badge">✦ AI Generated</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
  <p class="footer-text">
    Deep Learning
    <span class="footer-dot">✦</span>
    LSTM Architecture
    <span class="footer-dot">✦</span>
    Streamlit
  </p>
</div>
""", unsafe_allow_html=True)