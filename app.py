
#SportDigest AI — Démo Streamlit de résumé automatique d'articles sportifs français.

#Comparaison de 3 type de modele :
#1. mT5 vanilla (Google, pré-entraîné non spécialisé)
#2. mT5 fine-tuné sur MLSUM-sport (voir le notebook de fine-tuning pour les détails)
#3. CroissantLLM via prompting (LLM bilingue FR/EN)

import time
import streamlit as st
import pandas as pd
from datasets import load_dataset

from src.summarizer import MT5Summarizer, LLMSummarizer, DEFAULT_MODEL, FINETUNED_MODEL



# Configuration de la page

st.set_page_config(
    page_title="SportDigest AI",
    page_icon="🏟️",
    layout="wide",
)

st.title("🏟️ SportDigest AI")
st.markdown(
    "**Résumé automatique d'articles sportifs en français** — "
    "Comparaison entre un modèle multilingue généraliste, sa version fine-tunée sur "
    "MLSUM-sport, et un LLM bilingue via prompting."
)



# Chargement des modèles 


@st.cache_resource(show_spinner="Chargement de mT5 vanilla...")
def load_mt5_vanilla():
    return MT5Summarizer(model_name=DEFAULT_MODEL)

@st.cache_resource(show_spinner="Chargement de mT5 fine-tuné...")
def load_mt5_finetuned():
    return MT5Summarizer(model_name=FINETUNED_MODEL)

@st.cache_resource(show_spinner="Chargement de CroissantLLM (~2.5 Go, peut prendre 3-5 min)...")
def load_llm():
    return LLMSummarizer()



# Chargement d'un sous-échantillon du test set MLSUM-sport pour la démo (20 articles aléatoires)

SPORT_TOPICS = [
    "athletisme", "basket", "blog-du-tour-de-france", "blog-roland-garros",
    "championnats-monde-athletisme", "coupe-du-monde", "coupe-du-monde-rugby",
    "cyclisme", "football", "formule-1", "golf", "handball",
    "jeux-olympiques", "jeux-olympiques-pyeongchang-2018", "jeux-olympiques-rio-2016",
    "le-nouveau-roland-garros", "ligue-1", "ligue-des-champions",
    "natation", "roland-garros", "rugby", "ski", "sport",
    "tennis", "top-14", "tour-de-france", "voile",
]

@st.cache_data(show_spinner="Téléchargement d'un échantillon MLSUM-sport...")
def load_sample_articles(n: int = 20) -> pd.DataFrame:
    BASE_URL = "hf://datasets/reciTAL/mlsum@refs/convert/parquet/fr"
    ds = load_dataset("parquet", data_files={"test": f"{BASE_URL}/test/*.parquet"})
    df = ds["test"].to_pandas()
    df = df[df["topic"].isin(SPORT_TOPICS)].reset_index(drop=True)
    return df.sample(n=n, random_state=42).reset_index(drop=True)



# Sidebar avec choix des modèles à comparer

st.sidebar.header("⚙️ Configuration")

use_vanilla = st.sidebar.checkbox("mT5 vanilla (Google)", value=True)
use_finetuned = st.sidebar.checkbox("mT5 fine-tuné sport (notre modèle)", value=True)
use_llm = st.sidebar.checkbox(
    "CroissantLLM (1.3B, prompting)",
    value=False,
    help="Plus lent à charger (~2.5 Go), désactivé par défaut.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Modèles utilisés :**\n"
    f"- Vanilla : `{DEFAULT_MODEL}`\n"
    f"- Fine-tuné : `{FINETUNED_MODEL}`\n"
    "- LLM : `croissantllm/CroissantLLMChat-v0.1`"
)


# Onglets

tab1, tab2 = st.tabs([" Démo", " Résultats d'évaluation"])

#Onglet 1 : Démo 
with tab1:
    st.subheader("Choix de l'article")

    source = st.radio(
        "Source :",
        ["Article du test set MLSUM-sport", "Coller mon propre texte"],
        horizontal=True,
    )

    article_text = ""
    reference = None

    if source == "Article du test set MLSUM-sport":
        df_samples = load_sample_articles(n=20)
        labels = [f"[{r['topic']}] {r['title'][:80]}..." for _, r in df_samples.iterrows()]
        idx = st.selectbox("Sélectionne un article :", range(len(labels)), format_func=lambda i: labels[i])
        article_text = df_samples.iloc[idx]["text"]
        reference = df_samples.iloc[idx]["summary"]
        st.text_area("Article :", article_text, height=200, disabled=True)
    else:
        article_text = st.text_area(
            "Colle ton article ici :",
            height=200,
            placeholder="Article sportif en français...",
        )

    # Bouton de génération
    if st.button(" Générer les résumés", type="primary", use_container_width=True):
        if not article_text.strip():
            st.warning("Article vide.")
            st.stop()

        # Réf
        if reference:
            st.markdown("###  Résumé de référence (gold)")
            st.info(reference)

        # Colonnes pour les modèles
        cols = []
        active = []
        if use_vanilla: active.append("vanilla")
        if use_finetuned: active.append("finetuned")
        if use_llm: active.append("llm")

        if not active:
            st.warning("Active au moins un modèle dans la sidebar.")
            st.stop()

        cols = st.columns(len(active))

        # mT5 vanilla
        if use_vanilla:
            with cols[active.index("vanilla")]:
                st.markdown("#### 🔵 mT5 vanilla")
                with st.spinner("Génération..."):
                    model = load_mt5_vanilla()
                    t0 = time.time()
                    summary = model.summarize(article_text)
                    duration = time.time() - t0
                st.success(summary)
                st.caption(f"⏱️ {duration:.1f}s | {len(summary.split())} mots")

        # mT5 fine-tuné
        if use_finetuned:
            with cols[active.index("finetuned")]:
                st.markdown("#### 🟢 mT5 fine-tuné")
                with st.spinner("Génération..."):
                    model = load_mt5_finetuned()
                    t0 = time.time()
                    summary = model.summarize(article_text)
                    duration = time.time() - t0
                st.success(summary)
                st.caption(f"⏱️ {duration:.1f}s | {len(summary.split())} mots")

        # LLM
        if use_llm:
            with cols[active.index("llm")]:
                st.markdown("#### 🟠 CroissantLLM")
                with st.spinner("Génération..."):
                    model = load_llm()
                    t0 = time.time()
                    summary = model.summarize(article_text)
                    duration = time.time() - t0
                st.success(summary)
                st.caption(f"⏱️ {duration:.1f}s | {len(summary.split())} mots")


#Onglet 2 : Résultats d'évaluation 
with tab2:
    st.subheader("Évaluation comparative sur 100 articles MLSUM-sport (test)")

    # Tableau des scores (issus de la Session 5)
    df_rouge = pd.DataFrame({
        "ROUGE-1": [0.0103, 0.2718, 0.1954],
        "ROUGE-2": [0.0006, 0.1072, 0.0622],
        "ROUGE-L": [0.0088, 0.2096, 0.1312],
    }, index=["mT5 vanilla", "mT5 fine-tuné", "CroissantLLM"])

    st.dataframe(df_rouge.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"))

    st.bar_chart(df_rouge)

    st.markdown("""
    ### Interprétation

    - **mT5 vanilla** obtient des scores quasi nuls : sans fine-tuning, le modèle
      pré-entraîné de Google ne sait pas générer de résumés (il produit du bruit).

    - **mT5 fine-tuné** sur 5 000 articles MLSUM-sport gagne largement, avec
      un **ROUGE-1 multiplié par 26** par rapport au vanilla. Il dépasse meme le
      CroissantLLM (LLM 4× plus gros) sur toutes les métriques.

    - **CroissantLLM** produit des résumés français cohérents mais a tendance à
      paraphraser et halluciner, ce qui fait baisser les métriques ROUGE.

    **Conclusion** : la spécialisation par fine-tuning sur un domaine restreint
    permet à un modèle léger (300M params) de surpasser un LLM généraliste plus
    gros (1.3B params).
    """)



st.markdown("---")
st.caption(
    " Projet M2 Data Science & IA — NEXA (2025-2026) "
    "[Code source](https://github.com/Arcsin720/Sport-News)"
)