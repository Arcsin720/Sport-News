#  Sport_News IA

> **Comparaison de modèles de résumé automatique d'articles sportifs en français** — Projet réalisé dans le cadre du cours **Large Language Models** (M2 Data Science & IA, NEXA).

[Modèle fine-tuné hébergé sur HuggingFace] (https://huggingface.co/ArcSin720/mt5-sport-finetuned)

---

##  Problématique

**Le fine-tuning d'un modèle de langage multilingue généraliste sur un domaine restreint (sport en français) permet-il à un modèle léger de surpasser un LLM bilingue plus volumineux utilisé via prompting ?**

Pour répondre, nous comparons trois modele sur la tâche de résumé abstractif d'articles sportifs français :

1. **mT5-small vanilla** (Google) — modèle pré-entraîné multilingue, non spécialisé
2. **mT5-small fine-tuné** sur MLSUM-sport — (Travail sur le fine-tuning effectué)
3. **CroissantLLM-Chat** (1.3B params) — LLM bilingue via prompt engineering

---

##  Résultats principaux

Évaluation sur **100 articles** du test set MLSUM-sport  :

| Modèle | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| mT5 vanilla | 0.0103 | 0.0006 | 0.0088 |
| **mT5 fine-tuné** | **0.2718** | **0.1072** | **0.2096** |
| CroissantLLM | 0.1954 | 0.0622 | 0.1312 |

**Conclusions :**

- Le fine-tuning **multiplie ROUGE-1 par 26** par rapport au modèle vanilla
- Le mT5 fine-tuné (300M params) **est meilleur que CroissantLLM** (1.3B params, 4× plus gros) sur les trois métriques
- Démonstration de l'utilité de la spécialisation par fine-tuning sur un domaine restreint

---

## Structure du projet

sport-news/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploration MLSUM-FR + filtrage sport
│   ├── 02_baseline.ipynb         # Inférence mT5 vanilla + CroissantLLM
│   ├── 03_finetuning.ipynb       # Fine-tuning de mT5-small sur MLSUM-sport (Colab T4)
│   └── 04_evaluation.ipynb       # Évaluation ROUGE comparative
├── src/
│   ├── init.py
│   └── summarizer.py             # Classes MT5Summarizer et LLMSummarizer
├── app.py                        # Démo Streamlit 
├── requirements.txt
├── .gitignore
└── README.md

---

## Méthodologie

### 1. Dataset

[MLSUM-FR](https://huggingface.co/datasets/reciTAL/mlsum) (Multilingual Summarization, config française) — articles du quotidien **Le Monde** avec résumés de référence.

- Total brut d'articles  : 392 902 articles (split `train`)
- Filtrage sport : 28 024 articles (train: 26 119 | val: 808 | test: 1 097)
- 34 topics sportifs identifiés via mots-clés choisis, nettoyés manuellement (exclusion de faux positifs : `port-du-voile`, `transports`)

### 2. Modèles

| Modèle | Référence | Params | Rôle |
|---|---|---|---|
| `google/mt5-small` | mT5 (ICML 2021, Google) | 300M | Baseline vanilla |
| `ArcSin720/mt5-sport-finetuned` | Notre fine-tuning | 300M | Spécialisé sport FR |
| `croissantllm/CroissantLLMChat-v0.1` | CroissantLLM (INRIA/Centrale Lyon) | 1.3B | LLM via prompting |

### 3. Fine-tuning

- **Données** : 5 000 articles train + 500 val (sous-échantillon aléatoire de MLSUM-sport)
- **Configuration** : 2 epochs, batch effectif = 8, Adafactor, learning rate = 1e-3
- **Optimisations** : gradient checkpointing, bf16 et padding dynamique
- **Hardware** : Google Colab T4 (16 Go VRAM)
- **Temps** : environs 25 minutes
- **Évolution loss** : Amélioration globale -> sur le train 5.21 → 4.14 | sur la validation 2.46 → 2.23

### 4. Évaluation

- **Métriques** : ROUGE-1, ROUGE-2, ROUGE-L (F1, métrique de réference pour du résumé automatique)
- **Échantillon** : 100 articles test 
- **Implémentation** : `rouge_score` 

---

## Reproduction du projet

### Prérequis

- Python 3.10+
- 5 Go d'espace disque (modèles HuggingFace en cache)
- Pour le fine-tuning : Google Colab GPU T4 recommandé

### Installation

```bash
git clone https://github.com/Arcsin720/Sport-News.git
cd Sport-News
python -m venv .venv
source .venv/bin/activate    # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

### Lancer la démo Streamlit

```bash
streamlit run app.py
```

L'app se lance sur `http://localhost:8501`. Le modèle fine-tuné est téléchargé depuis HuggingFace Hub au premier lancement.

### Refaire le fine-tuning (optionnel)

Le notebook `notebooks/03_finetuning.ipynb` est exécutable directement sur Google Colab :

1. Ouvrir le notebook sur Colab
2. Activer le GPU T4 
3. Exécuter toutes les cellules 
4. Le modèle est sauvegardé localement, puis poussé sur HF Hub

### Refaire l'évaluation

```bash
# En local (CPU/MPS)
jupyter notebook notebooks/04_evaluation.ipynb

# Sur Colab 
# → ouvrir le notebook sur Colab
```

---

## Analyse critique

### Points forts du fine-tuning

- Gain sur toutes les métriques ROUGE
- Surpasse un LLM 4× plus volumineux
- Temps d'entraînement court (~25 min sur GPU)

### Limites observées

1. **bias sur les premiers mots** — Sur des articles hors-distribution (style narratif inhabituel), le modèle veux reprendre les premières phrases.

3. **Hallucinations du LLM** — CroissantLLM ajoute parfois des détails qui n'existe pas dans l'article (vu dans les exemples qualitatifs).

### Pistes d'amélioration

- Élargir le dataset à des sources encore plus variées (L'Équipe, RMC, magazines)
- Utiliser un modèle plus gros (mT5-base ou mT5-large) avec du gradient.
- Ajouter une evaluation complémentaire avec BERTScore  pour déterminer les similarités des mots


---

## technique

- **Python** 3.13 (local) / 3.12 (Colab)
- **PyTorch**  CUDA (Colab)
- **HuggingFace** Transformers, Datasets, Hub
- **Streamlit** pour la démo interactive
- **Matplotlib, Seaborn** pour les visualisations
- **rouge-score** pour l'évaluation

---

## Auteur

**Taoufik Asoufi** — M2 Data Science & Intelligence Artificielle — NEXA (2025-2026)

- Modèle publié : [ArcSin720/mt5-sport-finetuned](https://huggingface.co/ArcSin720/mt5-sport-finetuned)
- Code source : [github.com/Arcsin720/Sport-News](https://github.com/Arcsin720/Sport-News)

---

## Références

- **MLSUM** : Scialom et al. (2020) — *MLSUM: The Multilingual Summarization Corpus*
- **mT5** : Xue et al. (2021) — *mT5: A massively multilingual pre-trained text-to-text transformer* (ICML 2021)
- **CroissantLLM** : Faysse et al. (2024) — *CroissantLLM: A Truly Bilingual French-English Language Model*
- **ROUGE** : Lin (2004) — *ROUGE: A Package for Automatic Evaluation of Summaries*
