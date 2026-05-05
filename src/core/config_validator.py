"""
WHY THIS FILE EXISTS:
Validates the tokenization configuration for the GLM-4-Voice model.

WHY: Because mismatched tokenizer configurations between the server and the bridge 
will cause silent catastrophic failures during audio generation. We run this 
to ensure the special tokens actually map to the IDs the server expects.
"""
import os
import json
from transformers import AutoTokenizer, AutoConfig

tokenizer_path = "/app/glm/glm-4-voice-9b"
if not os.path.exists(tokenizer_path):
    tokenizer_path = "THUDM/glm-4-voice-9b"

try:
    config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Vocab Size: {config.vocab_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    special_tokens = [
        "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", 
        "streaming_transcription", "<|begin_of_audio|>", "<|end_of_audio|>"
    ]
    for t in special_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"{t}: {tid}")
        
    print(f"Pad Token ID: {tokenizer.pad_token_id}")
    print(f"EOS Token ID: {tokenizer.eos_token_id}")
    
except Exception as e:
    print(f"Error: {e}")
