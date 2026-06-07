import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM


# Modèle BARThez déjà fine-tuné 
# On part de ce modèle pour la baseline, et on le fine-tunera pour le sport egalement 
DEFAULT_MODEL = "moussaKam/barthez-orangesum-abstract"
# LLM open-source bilingue FR/EN, avec prompting
DEFAULT_LLM = "croissantllm/CroissantLLMChat-v0.1"


def get_device() -> str:

    #Détecte le meilleur device disponible. 

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BARThezSummarizer:
    
    # générer les résumés avec un modèle BARThez.

    #Utilisation :
    #    summarizer = BARThezSummarizer()
    #    summary = summarizer.summarize("Texte de l'article...")
   

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device or get_device()

        print(f" Chargement du modèle {model_name} sur {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f" Modèle chargé")

    def summarize(
        self,
        text: str,
        max_source_length: int = 1024,
        max_target_length: int = 64,
        num_beams: int = 4,
    ) -> str:
        
        #Génère un résumé pour un article donné.

        #Args:
        #    text: l'article à résumer
        #    max_source_length: longueur max de l'input (en tokens), tronqué si plus 
        #    max_target_length: longueur max du résumé généré (en tokens)
        #    num_beams: nombre de beams pour la beam search (qualité vs vitesse)

        #Retourne => Le résumé généré en français
        
        # Tokenisation de l'article (avec troncature si trop long)
        inputs = self.tokenizer(
            text,
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Génération avec beam search
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,  
                early_stopping=True,
            )
        # Décodage des tokens en texte
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary


class LLMSummarizer:

    # Prompt template — structuré comme un dialogue chat
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
        #    text: l'article à résumer
        #    max_article_chars: tronque l'article si trop long 
        #    max_new_tokens: longueur max du résumé généré
        #    temperature: faible = factuel, élevé = créatif. Pour un résumé, on veut bas car one ne veux pas qu'il invente des infos.

        #retourne => Le résumé généré en français
        
        # Troncature de l'article pour rester dans la fenêtre de contexte
        article_truncated = text[:max_article_chars]

        # Construction du prompt
        prompt = self.PROMPT_TEMPLATE.format(article=article_truncated)

        # Tokenisation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        # Génération
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Décodage : on ne garde que la partie générée (pas le prompt)
        generated_ids = output_ids[0][prompt_length:]
        summary = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return summary