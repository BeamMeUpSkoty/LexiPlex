import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaModel
from functools import lru_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ValenceArousalModel:
    def __init__(self, model_dir="models/XLM-RoBERTa-base-MSE", embed_device="cpu"):
        # Scoring model
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        # Backbone for embeddings
        self.backbone: XLMRobertaModel = self.model.roberta
        self.embed_device = embed_device
        self.model.to(embed_device)

    def batch_score(self, texts, batch_size=64, device="cpu"):
        """
        Returns Nx2 tensor of (valence, arousal) for each input text.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt",
                                    padding=True, truncation=True).to(device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            scaled = torch.nn.Hardtanh(-1,1)(logits)
            results.append(scaled.cpu())
        return torch.vstack(results)

    @lru_cache(maxsize=10_000)
    def sentence_score(self, sentence):
        return self.batch_score((sentence,), batch_size=1)[0]

    def embed_sentences(self, sentences, batch_size=32) -> np.ndarray:
        """
        Compute sentence embeddings (CLS token) using the RoBERTa backbone.
        Returns a NumPy array of shape (N, D).
        """
        all_embeds = []
        self.backbone.to(self.embed_device)
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt",
                                    padding=True, truncation=True,
                                    max_length=128).to(self.embed_device)
            with torch.no_grad():
                outputs = self.backbone(**inputs).last_hidden_state
                cls_embeds = outputs[:, 0, :]
                all_embeds.append(cls_embeds.cpu().numpy())
        return np.vstack(all_embeds)

    def compute_salience(self, sentences, top_k=None, threshold=None) -> np.ndarray:
        """
        Compute salience scores as cosine similarity of each sentence embedding
        to the centroid of all embeddings. Returns an array of scores.
        Optionally returns only indices above threshold or top_k.
        """
        embeds = self.embed_sentences(sentences)
        centroid = embeds.mean(axis=0, keepdims=True)
        sims = cosine_similarity(embeds, centroid).ravel()
        # Optionally filter
        if top_k is not None:
            idxs = np.argsort(sims)[-top_k:]
            mask = np.zeros_like(sims, dtype=bool)
            mask[idxs] = True
            return sims, mask
        if threshold is not None:
            mask = sims >= threshold
            return sims, mask
        return sims, np.ones_like(sims, dtype=bool)
