# IR Assignment 1 — Streamlit Application

**Course:** AIMLCZG537 / DSECLZG537 — Information Retrieval (S2 2025-26)

---

## Dependencies

Install with pip:

```bash
pip install -r requirements.txt
```

Python 3.9+ required.

---

## Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Features

| Tab | Component |
|-----|-----------|
| 📁 Upload & View | Load 15 sample IR documents or upload your own `.txt` files |
| ⚙️ Preprocessing | Step-by-step pipeline: tokenisation → lowercasing → stop word removal → hyphen handling → stemming/lemmatisation; inverted index builder; stemming vs lemmatisation comparison |
| 🔗 Phrase Query | Biword index and positional index phrase search; false positive demonstration; multi-query comparison table |
| 🌳 Dictionary Search | BST and B-Tree (t=3) built from the document vocabulary; single-term and 20-query benchmark with comparison counts and timing |
| 🛡️ Tolerant Retrieval | Wildcard (K-gram, k=2/3), spelling correction (edit distance), Levenshtein DP matrix, K-gram index explorer, Soundex phonetic matching |
| 📊 Inference & Discussion | Answers all 7 required inference questions with tables and conclusions |

---

## Dataset

The app ships with 15 built-in IR-related documents covering topics such as:
information retrieval, search engines, text preprocessing, BST, B-Trees,
NLP, machine learning, deep learning, tolerant retrieval, phrase indexing.

You may upload your own `.txt` files via the Upload tab.

---

## Notes

- NLTK data is downloaded automatically on first run (requires internet).
- All indexes are built in-memory per session; no external database needed.
- The B-Tree implementation uses minimum degree t=3 (nodes hold 2–5 keys).
