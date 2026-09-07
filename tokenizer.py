class CharTokenizer:
    def __init__(self, text):
        self.chars=sorted(list(set(text)))
        self.stoi={ch:i for i,ch in enumerate(self.chars)}
        self.itos={i:ch for i,ch in enumerate(self.chars)}
        self.vocab_size=len(self.chars)
    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])