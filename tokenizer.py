from nltk.tokenize import word_tokenize


class SimpleTokenizer:
    def __init__(self, text):
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.build_vocab(text)
    
    def build_vocab(self, text):
        tokens = word_tokenize(text)
        self.vocab = sorted(list(set(tokens)))
        self.vocab_size = len(self.vocab) + 2
        self.stoi = {w:i for i, w in enumerate(self.vocab, start=2)}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.itos = {i:w for w, i in self.stoi.items()}
    
    def encode(self, text):
        tokens = word_tokenize(text)
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]
    
    def decode(self, indices):
        return ' '.join([self.itos.get(i, '<unk>')] for i in indices)