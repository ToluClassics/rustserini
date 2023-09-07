import numpy as np
from transformers import AutoModel, AutoTokenizer

from pyserini.encode import DocumentEncoder


class AutoDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0', pooling='cls', l2_norm=False):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False)
        self.has_model = True
        self.pooling = pooling
        self.l2_norm = l2_norm

    def encode(self, texts, titles=None, max_length=256, add_sep=False, **kwargs):
        shared_tokenizer_kwargs = dict(
            max_length=max_length,
            truncation=True,
            padding='longest',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
            add_special_tokens=True,
        )
        input_kwargs = {}
        if not add_sep:
            input_kwargs["text"] = [f'{title} {text}' for title, text in zip(titles, texts)] if titles is not None else texts
        else:
            if titles is not None:
                input_kwargs["text"] = titles
                input_kwargs["text_pair"] = texts
            else:
                input_kwargs["text"] = texts

        inputs = self.tokenizer(**input_kwargs, **shared_tokenizer_kwargs)
        inputs.to(self.device)
        outputs = self.model(**inputs)
        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()
        else:
            embeddings = outputs[0][:, 0].detach().cpu().numpy()

        return embeddings


if __name__ == '__main__':
    encoder = AutoDocumentEncoder(model_name='bert-base-uncased', device='cpu')
    queries = ['Title 1 Hello, I am a sentence!', 'Title 2 And another sentence.']
    query_embeddings = encoder.encode(queries, max_length=128)
    print(query_embeddings[0][:10])
    print(query_embeddings[0][:10])
