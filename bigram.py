import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
torch.manual_seed(42)
block_size = 8  # context length
batch_size = 32  # how many sequences per batch
epochs = 60000
lr = 1e-3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# data
with open("data/harry_potter.txt", "r") as f:
    data = f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

charidx = {ch: i for i, ch in enumerate(chars)}
idxchar = {i: ch for i, ch in enumerate(chars)}
encode = lambda str: [charidx[ch] for ch in str]
decode = lambda idx: "".join([idxchar[i] for i in idx])

data = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ids = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[idx : idx + block_size] for idx in ids])
    yb = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in ids])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )  # read of logits from lookup table (token-wise)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # shape: (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx)  # (B,T,C)
            logits = logits[:, -1, :]  # last time step only -> (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            pred = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, pred), dim=1)  # (B,T+1)
        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for _ in range(epochs):
    xb, yb = get_batch("train")
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

# test
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))
