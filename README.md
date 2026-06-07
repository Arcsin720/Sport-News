# Résumé article sportif AI

 Système de résumé automatique d'articles sportifs français basé sur des modèles seq2seq pré-entraînés et fine-tunés.

## Contexte

Projet réalisé dans le cadre du cours **Large Language Models** (M2 Data Science & IA, NEXA).

## Problématique

Le fine-tuning d'un modèle pré-entraîné de résumé français (BARThez) sur le sous-domaine sport améliore-t-il la qualité des résumés générés, mesurée par les métriques ROUGE ?

## Approche

Comparaison de trois modèles sur le dataset **MLSUM-FR** filtré sur les topics sportifs :

| Modèle | Type | Rôle |
|---|---|---|
| `barthez-orangesum-abstract` | Pré-entraîné (FR, résumé) | Baseline |
| `barthez-sport` (fine-tuné) | Spécialisé sport | Contribution |

Évaluation par métriques **ROUGE-1, ROUGE-2, ROUGE-L**.

## 📂 Structure du projet

## Installation

```bash
git clone <repo-url>
cd sport-news
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

##

Taoufik Asoufi — M2 Data Science & IA — NEXA (2025-2026)



