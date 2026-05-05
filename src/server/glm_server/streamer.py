import torch
from queue import Queue
from transformers.generation.streamers import BaseStreamer
from transformers import LogitsProcessor

class TokenQueueStreamer(BaseStreamer):
    """Bridge between model generation thread and FastAPI streaming response."""
    def __init__(self, prompt_len=0):
        self.q = Queue()
        self.stop_signal = object()
        self.prompt_len = prompt_len
        self.token_count = 0
        self.all_tokens = []

    def put(self, value):
        # Value can be a single ID or a tensor of multiple IDs
        tokens = value.view(-1)
        for t in tokens:
            tid = t.item()
            self.q.put(tid)
            self.all_tokens.append(tid)

    def end(self):
        self.q.put(self.stop_signal)

    def __iter__(self): return self
    def __next__(self):
        val = self.q.get()
        if val is self.stop_signal: raise StopIteration
        return val

class VocabGuardProcessor(LogitsProcessor):
    """Prevents model from generating tokens outside the valid embedding matrix."""
    def __init__(self, true_vocab_size):
        self.true_vocab_size = true_vocab_size

    def __call__(self, input_ids, scores):
        if scores.shape[-1] > self.true_vocab_size:
            scores[:, self.true_vocab_size:] = -float("inf")
        return scores
