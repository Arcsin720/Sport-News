
#Module de résumé d'articles sportifs en français.

#2 Modele a utiliser  :
# MT5Summarizer : mT5 multilingue fine-tuné sur XL-Sum (baseline + cible du fine-tuning)
# LLMSummarizer : CroissantLLM via prompting (LLM bilingue FR/EN open-source)


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# Modèle de résumé multilingue, pré-entraîné sur XL-Sum (BBC News, 45 langues)
# Paper : ACL 2021 — csebuetnlp (Bangladesh University of Eng. and Tech.)
DEFAULT_MODEL = "csebuetnlp/mT5_multilingual_XLSum"

# LLM open-source bilingue FR/EN, instruction-tuned, léger (1.3B)
DEFAULT_LLM = "croissantllm/CroissantLLMChat-v0.1"


def get_device() -> str:
    """Détecte le meilleur device disponible (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MT5Summarizer:
    
    #Résumeur basé sur mT5 multilingue .
    #Sert à la fois de baseline et de modèle à fine-tuner sur MLSUM-sport.
    

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device or get_device()

        print(f"Chargement du modèle {model_name} sur {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Modèle chargé")

    def summarize(
        self,
        text: str,
        max_source_length: int = 512,
        max_target_length: int = 64,
        num_beams: int = 4,
    ) -> str:
        """Génère un résumé pour un article donné."""
        inputs = self.tokenizer(
            text,
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary


class LLMSummarizer:
    #Résumeur basé sur CroissantLLM (1.3B de parametre) avec prompting

    PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "Tu es un journaliste sportif francophone. "
        "Tu rédiges des résumés courts, factuels et précis, en français, "
        "en 2 à 3 phrases maximum.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Résume l'article de presse suivant :\n\n{article}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(self, model_name: str = DEFAULT_LLM, device: str = None):
        self.model_name = model_name
        self.device = device or get_device()

        print(f"Chargement du LLM {model_name} sur {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print(f"LLM chargé")

    def summarize(
        self,
        text: str,
        max_article_chars: int = 4000,
        max_new_tokens: int = 150,
        temperature: float = 0.3,
    ) -> str:
        #Génère un résumé via prompting.
        article_truncated = text[:max_article_chars]
        prompt = self.PROMPT_TEMPLATE.format(article=article_truncated)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_length:]
        summary = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return summary