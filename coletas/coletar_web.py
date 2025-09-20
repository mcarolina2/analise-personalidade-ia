import praw
import pandas as pd
import re
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

reddit = praw.Reddit(
    client_id="9eKySdLoUGdwX3RvmfyRrA",
    client_secret="u3nKPdIlqv-79vYbKXWn3ttO5zw6nQ",
    user_agent="analise-personalidade-ia:v1.0"
)

subreddits = ["ChatGPT", "artificial", "MachineLearning"]

regex_ia = re.compile(r"\b(ai|artificial intelligence|chatgpt|gpt|machine learning|llm|language model|inteligência artificial)\b", re.IGNORECASE)

comentarios_lista = []

for subreddit_name in subreddits:
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=50):
            if post.stickied:
                continue
            try:
                post.comments.replace_more(limit=0)
            except:
                continue

            for comment in post.comments.list():
                if hasattr(comment, 'body') and regex_ia.search(comment.body):
                    comentarios_lista.append({
                        "subreddit": subreddit_name,
                        "post_title": post.title,
                        "comment": comment.body[:200]+"..." if len(comment.body) > 200 else comment.body,
                        "upvotes": comment.score
                    })
    except Exception as e:
        print(f"Erro em r/{subreddit_name}: {e}")

# Salvar CSV
df = pd.DataFrame(comentarios_lista)
df.to_csv("../dados/comentarios_ia_bruto.csv", index=False, encoding="utf-8")
print(f"Salvos {len(df)} comentários no CSV")

