import re
import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk 

nltk.download("vader_lexicon")

# ----------------------------
# Config
# ----------------------------
INPUT_CSV = "../dados/comentarios_ia_bruto.csv"            
OUTPUT_CSV = "../saidas/Output_reddit_CSV.csv"
SPACY_MODEL = "en_core_web_sm"    

sia = SentimentIntensityAnalyzer()

# termos IA
ia_terms = [
    "ai", "artificial intelligence", "inteligencia artificial", "inteligência artificial",
    "chatgpt", "gpt", "llm", "language model", "machine learning"
]
pattern_ia = re.compile(r"\b(?:{})\b".format("|".join(re.escape(t) for t in ia_terms)),
                        flags=re.IGNORECASE)


# ----------------------------
# Função de sentimento
# ----------------------------
def analisar_sentimento(texto: str) -> str:
    scores = sia.polarity_scores(texto)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positivo"
    elif compound <= -0.05:
        return "negativo"
    else:
        return "neutro"

# ----------------------------
# Funções
# ----------------------------
def clean_text(text: str) -> str:
    """Limpa links, menções e caracteres estranhos."""
    if not isinstance(text, str):
        return ""
    s = text.replace("\n", " ").strip()
    s = re.sub(r"http\S+", " ", s)           # URLs
    s = re.sub(r"www\.\S+", " ", s)
    s = re.sub(r"u\/\w+", " ", s)            # reddit user
    s = re.sub(r"\/r\/\w+", " ", s)          # reddit subreddit
    s = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\?\!\.,;:\-\'\"\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mentions_ia(text: str) -> bool:
    return bool(pattern_ia.search(text))


# ----------------------------
# Carregar modelo spaCy
# ----------------------------
nlp = spacy.load(SPACY_MODEL, disable=["ner"])
nlp.max_length = 2000000

# ----------------------------
# Ler CSV e processar
# ----------------------------
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

df["cleaned"] = df["comment"].astype(str).apply(clean_text)
df["mentions_ia"] = df["cleaned"].apply(mentions_ia)

# Filtrar comentários que mencionam IA
df_filtered = df[df["mentions_ia"]].reset_index(drop=True)
print(f"Total: {len(df)} — com IA: {len(df_filtered)}")

# ----------------------------
# Processamento NLP
# ----------------------------
rows_out = []

for doc, (_, row) in zip(nlp.pipe(df_filtered["cleaned"], batch_size=50), df_filtered.iterrows()):
    # tokens com POS
    tokens_info = [(t.text, t.pos_, t.dep_) for t in doc]

    # chunks (shallow parsing)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # pares substantivo-adjetivo (dependency parsing)
    adj_noun_pairs = []
    for tok in doc:
        if tok.pos_ == "ADJ":
            if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
                adj_noun_pairs.append((tok.head.text, tok.text))
            else:
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubj:pass") and child.pos_ in ("NOUN", "PROPN"):
                        adj_noun_pairs.append((child.text, tok.text))

    rows_out.append({
        "subreddit": row["subreddit"],
        "post_title": row["post_title"],
        "comment": row["comment"],
        "upvotes": row["upvotes"],
        "cleaned": row["cleaned"],
        "tokens_pos": str(tokens_info),
        "noun_chunks": "; ".join(noun_chunks),
        "adj_noun_pairs": str(adj_noun_pairs),
        "sentimento": analisar_sentimento(row["cleaned"])
    })

# ----------------------------
# Salvar resultado
# ----------------------------

df_out = pd.DataFrame(rows_out)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Processamento salvo em {OUTPUT_CSV}")
