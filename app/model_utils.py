import torch
import torch.nn as nn


# ===== MODEL CLASS (must match training exactly) =====
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, src, hidden=None):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return h0, c0


# ===== SIMPLE VOCAB =====
class SimpleVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.default_index = self.stoi.get("<unk>", 0)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def get_itos(self):
        return self.itos


# ===== LOAD MODEL =====
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Load pure state_dict
    state_dict = torch.load("../lotr_lstm_lm.pt", map_location=device)

    # 🔹 RECREATE vocab (must match training)
    # Replace this list with the SAME vocab used in training
    
    # Dummy vocab recreated to match training size
    VOCAB_SIZE = 884
    itos = [f"token_{i}" for i in range(VOCAB_SIZE)]
    vocab = SimpleVocab(itos)


    model = LSTMLanguageModel(
        vocab_size=len(vocab.get_itos()),
        emb_dim=512,
        hid_dim=512,
        num_layers=2,
        dropout_rate=0.65
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, vocab, device


# ===== TEXT GENERATION =====
def generate_text(prompt, model, vocab, device, max_len=30, temperature=0.8):
    tokens = prompt.lower().split()
    input_ids = torch.tensor([[vocab[t] for t in tokens]], device=device)

    hidden = model.init_hidden(1, device)
    generated = tokens.copy()

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(input_ids, hidden)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        next_word = vocab.get_itos()[next_id]
        generated.append(next_word)
        input_ids = torch.tensor([[next_id]], device=device)

        if next_word == "<eos>":
            break

    return " ".join(generated)
