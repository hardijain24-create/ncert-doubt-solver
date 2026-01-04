import streamlit as st
import os
import PyPDF2
import pickle
import hashlib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from textwrap import wrap

# =========================
# CONFIG
# =========================
BASE_PDF_DIR = "backend_pdfs"
CACHE_DIR = "vector_cache"
CHUNK_SIZE = 220
TOP_K = 6
TFIDF_WEIGHT = 0.4
EMBEDDING_WEIGHT = 0.6

os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(
    page_title="📘 NCERT Doubt Solver",
    layout="centered"
)

# =========================
# LOAD EMBEDDING MODEL
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# TEXT HELPERS
# =========================
def clean_text(text):
    return " ".join(text.replace("\n", " ").split())

def chunk_text(text, page, source):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if len(chunk) > 120:
            chunks.append({
                "text": chunk,
                "page": page,
                "source": source
            })
    return chunks

def compute_content_hash(chunks):
    h = hashlib.md5()
    for c in chunks:
        h.update(c["text"].encode())
    return h.hexdigest()

# =========================
# BUILD KNOWLEDGE BASE
# =========================
def build_ncert_index(grade, language, extra_chunks=None):
    chunks = []

    folder = os.path.join(BASE_PDF_DIR, f"Grade{grade}", language)
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(".pdf"):
                reader = PyPDF2.PdfReader(open(os.path.join(folder, f), "rb"))
                for i, p in enumerate(reader.pages):
                    t = p.extract_text()
                    if t:
                        chunks.extend(chunk_text(clean_text(t), i + 1, f))

    if extra_chunks:
        chunks.extend(extra_chunks)

    texts = [c["text"] for c in chunks]
    if not texts:
        return [], None, None, None

    vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    content_hash = compute_content_hash(chunks)
    cache_file = os.path.join(
        CACHE_DIR, f"{grade}_{language}_{content_hash}.pkl"
    )

    if os.path.exists(cache_file):
        embeddings = pickle.load(open(cache_file, "rb"))
    else:
        model = load_embedding_model()
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
        pickle.dump(embeddings, open(cache_file, "wb"))

    return chunks, vectorizer, tfidf_matrix, embeddings

# =========================
# HYBRID SEARCH
# =========================
def retrieve_chunks(question, chunks, vec, tfidf, emb):
    q_tfidf = vec.transform([question])
    tfidf_scores = cosine_similarity(q_tfidf, tfidf)[0]

    model = load_embedding_model()
    q_emb = model.encode([question])[0]
    emb_scores = cosine_similarity([q_emb], emb)[0]

    scores = TFIDF_WEIGHT * tfidf_scores + EMBEDDING_WEIGHT * emb_scores
    idx = scores.argsort()[-TOP_K:][::-1]

    return [chunks[i] for i in idx]

# =========================
# 7-PARAGRAPH NCERT ANSWER
# =========================
def build_long_ncert_answer(retrieved, grade, question):
    base_text = " ".join(r["text"] for r in retrieved)

    sentences = [
        s.strip()
        for s in base_text.replace("\n", " ").split(".")
        if 50 < len(s.strip()) < 300
    ]

    if len(sentences) < 5:
        return "⚠️ Sufficient NCERT explanation was not found for this question."

    model = load_embedding_model()
    sent_embs = model.encode(sentences)

    unique, unique_embs = [], []
    for s, e in zip(sentences, sent_embs):
        if all(cosine_similarity([e], [ue])[0][0] < 0.78 for ue in unique_embs):
            unique.append(s)
            unique_embs.append(e)

    while len(unique) < 12:
        unique.append(unique[-1])

    def para(a, b, c=None):
        text = f"{a}. {b}."
        if c:
            text += f" {c}."
        return text

    paragraphs = [
        para(
            unique[0],
            unique[1],
            f"In NCERT Class {grade} Science, this topic is introduced to help students clearly understand the basic idea behind the concept"
        ),

        para(
            unique[2],
            unique[3],
            "This explanation makes it easier for students to remember and reproduce the answer in examinations"
        ),

        para(
            unique[4],
            unique[5],
            "NCERT presents this concept in a step-by-step manner so that learners can understand how it works"
        ),

        para(
            unique[6],
            unique[7],
            "Such classification or structural explanation helps in better conceptual clarity"
        ),

        para(
            unique[8],
            unique[9],
            "This concept has significant importance in understanding related topics in the chapter"
        ),

        para(
            unique[10],
            unique[11],
            "These examples help students connect textbook knowledge with real-life situations"
        ),

        (
            f"In conclusion, {unique[0].lower().rstrip('.')}. "
            f"This topic is important from an examination point of view and should be revised thoroughly."
        )
    ]

    return "\n\n".join(paragraphs)

# =========================
# PDF EXPORT
# =========================
def generate_pdf(question, answer, grade):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, f"NCERT Long Answer – Class {grade}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Question:")
    y -= 18

    c.setFont("Helvetica", 11)
    for line in wrap(question, 90):
        c.drawString(x, y, line)
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Answer:")
    y -= 18

    c.setFont("Helvetica", 11)
    for para in answer.split("\n\n"):
        for line in wrap(para, 95):
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50
            c.drawString(x, y, line)
            y -= 14
        y -= 10

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# =========================
# MCQ GENERATOR
# =========================
def generate_mcqs(retrieved, num_mcqs=5):
    sentences = []
    for r in retrieved:
        for s in r["text"].split("."):
            s = s.strip()
            if 70 < len(s) < 200:
                sentences.append(s)

    mcqs = []
    used = set()

    for s in sentences:
        words = s.split()
        if len(words) < 8:
            continue

        key_index = len(words) // 2
        answer = words[key_index]

        if answer.lower() in used:
            continue
        used.add(answer.lower())

        question = s.replace(answer, "_____")

        distractors = [
            w for w in words
            if w.isalpha() and w.lower() != answer.lower()
        ]

        distractors = list(dict.fromkeys(distractors))[:3]
        options = distractors + [answer]

        if len(options) < 4:
            continue

        np.random.shuffle(options)

        mcqs.append({
            "question": question,
            "options": options,
            "answer": answer
        })

        if len(mcqs) == num_mcqs:
            break

    return mcqs

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎓 Study Settings")
grade = st.sidebar.selectbox("Class", [6,7,8,9,10])
language = st.sidebar.selectbox("Language", ["English","Hindi"])

uploads = st.sidebar.file_uploader(
    "📎 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

def parse_uploads(files):
    chunks = []
    for f in files:
        reader = PyPDF2.PdfReader(f)
        for i,p in enumerate(reader.pages):
            t = p.extract_text()
            if t:
                chunks.extend(chunk_text(clean_text(t), i+1, f.name))
    return chunks

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = []

if uploads:
    st.session_state.uploaded_chunks = parse_uploads(uploads)
    st.sidebar.success("✅ PDFs added")

# =========================
# BUILD KB
# =========================
if "ready" not in st.session_state:
    st.session_state.ready = False

if not st.session_state.ready:
    st.warning("⚠️ Please build the NCERT knowledge base first.")
    if st.button("📚 Build Knowledge Base"):
        with st.spinner("Preparing NCERT knowledge..."):
            c,v,t,e = build_ncert_index(
                grade, language, st.session_state.uploaded_chunks
            )
            st.session_state.chunks = c
            st.session_state.vec = v
            st.session_state.tfidf = t
            st.session_state.emb = e
            st.session_state.ready = True
        st.success("✅ Ready to study!")
        st.rerun()
    st.stop()

# =========================
# MAIN UI
# =========================
st.title("📘 NCERT Doubt Solver")
st.caption("NCERT-only | 7-paragraph | Exam-ready")

question = st.chat_input("Ask your NCERT question...")

if question:
    retrieved = retrieve_chunks(
        question,
        st.session_state.chunks,
        st.session_state.vec,
        st.session_state.tfidf,
        st.session_state.emb
    )

    answer = build_long_ncert_answer(retrieved, grade, question)

    st.markdown("## 📖 Answer")
    st.write(answer)

    pdf = generate_pdf(question, answer, grade)
    st.download_button(
        "📄 Download Answer as PDF",
        pdf,
        file_name="NCERT_Long_Answer.pdf",
        mime="application/pdf"
    )

    st.markdown("## 🧪 NCERT MCQs")
    mcqs = generate_mcqs(retrieved)

    for i, m in enumerate(mcqs, 1):
        st.write(f"**Q{i}. {m['question']}**")
        for opt in m["options"]:
            st.write(f"- {opt}")
        st.write(f"✅ Correct answer: **{m['answer']}**")
        st.write("---")

    st.markdown("## 📚 Sources")
    for r in retrieved:
        st.write(f"• Page {r['page']} | {r['source']}")
