import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.encoders import TextEncoder

device = 'cuda:0'
model = 'llama'
text_encoder = TextEncoder(model_type=model, layer='last').to(device) #penultimate

ego4d_verbs = np.load('Dataset/ego_4d_verb.npy')
text_embedding = []

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for text in ego4d_verbs:
    # get_embedding
    encoded_text = text_encoder(text)
    text_embedding.append(encoded_text.detach().cpu())

text_embedding = torch.stack(text_embedding).numpy().squeeze()
print(text_embedding.shape)

X_emb = TSNE(n_components=2, perplexity=10, n_iter=1000, verbose=False).fit_transform(text_embedding)  # returns shape (n_samples, 2)
ax.scatter(X_emb[:, 0], X_emb[:, 1])
for i, txt in enumerate(ego4d_verbs):
    ax.annotate(str(txt), (X_emb[i, 0]+np.random.randn()*10, X_emb[i, 1]+np.random.randn()*50))
plt.show()



