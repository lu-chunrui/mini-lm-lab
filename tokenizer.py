class CharTokenizer:
    def __init__(self, text=None,chars=None):
        if chars is not None:
            self.chars = chars
        elif text is not None:
            self.chars = sorted(list(set(text)))
        else:
            raise ValueError("必须提供text或chars")
        self.stoi={ch:i for i,ch in enumerate(self.chars)}
        self.itos={i:ch for i,ch in enumerate(self.chars)}
        self.vocab_size=len(self.chars)
    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])