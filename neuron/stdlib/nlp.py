"""Neuron NLP Library — Natural Language Processing"""
import numpy as np
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor

class Tokenizer:
    def __init__(self, mode='word'):
        self.mode = mode
        self.vocab = {}; self.inverse_vocab = {}; self.next_id = 0
        for special in ['<pad>', '<unk>', '<sos>', '<eos>']:
            self.vocab[special] = self.next_id
            self.inverse_vocab[self.next_id] = special
            self.next_id += 1
    def fit(self, texts):
        for text in texts:
            for token in self._split(text):
                if token not in self.vocab:
                    self.vocab[token] = self.next_id
                    self.inverse_vocab[self.next_id] = token
                    self.next_id += 1
        return self
    def encode(self, text):
        return [self.vocab.get(t, self.vocab['<unk>']) for t in self._split(text)]
    def decode(self, ids):
        return ' '.join(self.inverse_vocab.get(i, '<unk>') for i in ids)
    def _split(self, text):
        if self.mode == 'char': return list(text)
        return text.lower().split()
    @property
    def vocab_size(self): return len(self.vocab)

def pad_sequences(sequences, max_len=None, pad_value=0):
    if max_len is None: max_len = max(len(s) for s in sequences)
    result = []
    for seq in sequences:
        if len(seq) >= max_len: result.append(seq[:max_len])
        else: result.append(seq + [pad_value] * (max_len - len(seq)))
    return result

def bag_of_words(texts, vocab=None):
    if vocab is None:
        vocab = {}; idx = 0
        for text in texts:
            for word in text.lower().split():
                if word not in vocab: vocab[word] = idx; idx += 1
    result = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            if word in vocab: result[i, vocab[word]] += 1
    return NeuronTensor(result)

def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model//2] if d_model % 2 else pos * div)
    return NeuronTensor(pe)

def create_nlp_module():
    bf = BuiltinFunction
    attrs = {
        'Tokenizer': bf('Tokenizer', lambda a, **kw: Tokenizer(a[0] if a else 'word')),
        'pad_sequences': bf('pad_sequences', lambda a, **kw: pad_sequences(a[0], kw.get('max_len', None))),
        'bag_of_words': bf('bag_of_words', lambda a, **kw: bag_of_words(a[0])),
        'positional_encoding': bf('positional_encoding', lambda a, **kw: positional_encoding(int(a[0]), int(a[1]))),
        'lowercase': bf('lowercase', lambda a, **kw: a[0].lower()),
        'remove_punctuation': bf('remove_punctuation', lambda a, **kw: ''.join(c for c in a[0] if c.isalnum() or c.isspace())),
        'word_count': bf('word_count', lambda a, **kw: len(a[0].split())),
    }
    return NeuronModule("nlp", attrs)
