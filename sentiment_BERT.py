#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# %%
tokenizer =AutoTokenizer.\
    from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model=AutoModelForSequenceClassification.\
    from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# %%
# try
tokens=tokenizer.encode('Not so bad, I thought I would be better', return_tensors='pt')
# %%
result=model(tokens)
result.logits
int(torch.argmax(result.logits))+1
# %%
