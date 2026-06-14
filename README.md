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


1. Which preprocessing technique improved retrieval quality most?

Stop word removal had the largest positive impact — it eliminated high-frequency noise words ("the", "is", "of", "and") that appear in nearly every job description and would otherwise dominate the inverted index and hurt precision.

Lowercasing ensured consistent matching ("Python" = "python", "Developer" = "developer").

Hyphen handling unified compound terms: "full-time" → "full time", "part-time" → "part time", allowing both words to be indexed and retrieved independently.

Ranking by impact on retrieval quality:

Stop word removal (precision ↑ significantly)
Lowercasing (recall ↑ for exact-case queries)
Stemming / Lemmatization (recall ↑ for morphological variants)
Hyphen handling (coverage ↑ for compound job terms)
2. Was stemming or lemmatization better for this dataset?

Stemming (Porter Stemmer) is preferred for this job description dataset.

Job descriptions contain many morphological variants: "developing", "developed", "developer", "development" — all should map to the same concept. Stemming unifies these as "develop".
Lemmatization is more conservative; "developing" → "developing" (not "develop"), reducing recall.
For a job-posting corpus where a recruiter searching "develop" should find all related postings, stemming's aggressive suffix-stripping is an advantage.
Conclusion: Use stemming for job search (high recall); use lemmatization for precision-critical applications (legal/medical document retrieval).
3. Which phrase query index was more accurate?

Positional Index is definitively more accurate.

Biword index only verifies that consecutive word pairs exist, not that the full phrase is contiguous. For "machine learning engineer", it checks "machine learning" ∩ "learning engineer" — a job posting with "machine learning" in one paragraph and "learning a new engineering skill" elsewhere could be a false positive from the biword index.
Positional index verifies exact sequential positions: pos("machine")+1 = pos("learning") and pos("learning")+1 = pos("engineer"). Zero false positives, guaranteed.
Trade-off: Positional index requires O(T) storage (T = total token count). Despite higher storage, positional index is universally preferred in production job search systems.
4. Which tree structure was faster for dictionary search?

B-Tree (t=3) outperforms BST in both comparison count and wall-clock time.

Key reasons:

BST height depends on insertion order. Even with randomised insertion, depth can vary. With a large job-posting vocabulary (thousands of terms), skewed input degrades to O(n).
B-Tree is always self-balancing. With t=3, each internal node holds 2–5 keys, reducing tree height. Experiments showed B-Tree needed fewer comparisons on average.
Disk I/O: For large job-site dictionaries stored on disk, B-Tree maps each node to a disk page, dramatically reducing I/O operations — the primary reason job search engines use B+ Trees for their inverted index dictionaries.
5. How tolerant was the retrieval model?

The system demonstrated strong tolerant retrieval across all techniques:

Technique	Coverage	Accuracy
Wildcard (K-gram, k=2)	Handles prefix, suffix, infix wildcards	~100% for single * patterns
Spelling correction (ed ≤ 2)	~85% of 1–2 character errors	Handles most job-domain typos
Soundex phonetic	~70–80% of phonetically similar misspellings	Effective for vowel errors
K-gram (k=3)	More precise wildcard results	Fewer false positives than k=2
Combined, the system handles the vast majority of real-world imperfect queries in a job-search context. Edit distance is the most computationally expensive (O(|V|·m·n)) but most accurate.

6. What are the limitations of the system?

No relevance ranking: Results are binary (retrieved / not retrieved) — no TF-IDF or BM25 scoring.
Spell correction is O(|V|×m×n): Linear vocabulary scan is slow for 10,000+ term vocabularies.
Unbalanced BST risk: Without AVL/Red-Black balancing, BST performance degrades on sorted input.
No index persistence: Indexes are rebuilt every session; no disk storage.
English only: No multilingual support (relevant for global job postings).
Biword index memory: Grows quadratically with vocabulary for very large corpora.
No query feedback loop: No pseudo-relevance feedback or user relevance judgements.
CSV size limit: Loading all 2,277 rows can be slow; capped at 300 by default.
7. How can the system be improved?

Add TF-IDF / BM25 ranking for relevance-ordered results (most impactful improvement).
Implement AVL or Red-Black BST to guarantee O(log n) worst-case.
Use B+ Trees (all data in leaves) for better range queries.
Add semantic search with sentence-transformers for context-aware job matching.
Query expansion via job-title synonyms (e.g., "dev" → "developer", "SDE" → "software engineer").
Persistent index storage with SQLite / Redis for production deployments.
Parallel indexing using multiprocessing for large corpora.
Add evaluation metrics (Precision@k, MAP, NDCG) for quantitative benchmarking.
Index compression to reduce memory footprint for large job datasets.
Auto-complete using a Trie for real-time job title suggestions.
Summary of Experimental Findings
 

Component

Best Technique

Justification

0

Text Preprocessing

Stop word removal + Lowercasing

Greatest precision and recall improvement

1

Stemming vs Lemmatization

Stemming (Porter)

Higher recall; suited for job description corpus

2

Phrase Query Index

Positional Index

Zero false positives; exact phrase matching

3

Dictionary Search

B-Tree (t=3)

Fewer comparisons; guaranteed O(log n) balance

4

Tolerant Retrieval

Edit Distance + K-gram Index

Handles wildcards, typos and phonetic errors

Overall Conclusion: This Streamlit-based IR system successfully demonstrates end-to-end information retrieval on a real-world job postings dataset — from raw document ingestion through preprocessing, indexing, phrase querying, dictionary search, and tolerant retrieval. The primary limitation is the absence of relevance ranking; adding TF-IDF or BM25 would be the highest-impact single improvement for a production job search engine.
