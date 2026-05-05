import torch
import torchaudio
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bridge.glm_server.engine import GLMVoiceEngine
from bridge.glm_server.config import Colors

def calculate_similarity(v1, v2):
    """Cosine similarity between two embedding tensors."""
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)
    return torch.mm(v1, v2.t()).item()

def main():
    parser = argparse.ArgumentParser(description="Voice Cloning Fidelity Test")
    parser.add_argument("reference_audio", type=str, help="Path to a 5-10s reference WAV/MP3")
    parser.add_argument("--text", type=str, default="This is a test of the emergency voice cloning system. Do I sound like you?", help="Text to synthesize")
    args = parser.parse_args()

    if not os.path.exists(args.reference_audio):
        print(f"{Colors.RED}Error: Reference audio '{args.reference_audio}' not found.{Colors.RESET}")
        return

    print(f"{Colors.CYAN}Initializing GLM-4-Voice Engine (this may take a moment)...{Colors.RESET}")
    # Using default paths as per the environment
    engine = GLMVoiceEngine(
        model_path="THUDM/glm-4-voice-9b",
        tokenizer_path="THUDM/glm-4-voice-tokenizer",
        flow_path="/app/glm/glm-4-voice-decoder"
    )

    # 1. Profile the reference voice
    print(f"\n{Colors.YELLOW}Step 1: Profiling reference audio...{Colors.RESET}")
    audio, sr = torchaudio.load(args.reference_audio)
    
    # Convert to 16-bit Mono PCM for the engine
    # (Matches Discord bridge behavior)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio_int16 = (audio * 32767).to(torch.int16)
    pcm_bytes = audio_int16.numpy().tobytes()
    pcm_b64 = base64.b64encode(pcm_bytes).decode()
    
    # We provide a dummy prompt text for the test if none is provided
    # to exercise the grounding logic.
    prompt_text = "This is the reference audio sample used for cloning."
    
    engine.set_voice_reference(pcm_b64, "test_subject", sample_rate=sr, prompt_text=prompt_text)
    ref_embedding = engine.ref_prompt_embedding
    
    if ref_embedding is None:
        print(f"{Colors.RED}Error: Failed to extract speaker embedding. Is SpeakerEncoder available?{Colors.RESET}")
        return

    # Use a longer sequence (200 tokens ≈ 2s) to capture more identity nuances
    test_tokens = engine.ref_prompt_token[:, :200] 
    
    print(f"\n{Colors.YELLOW}Step 2: Synthesizing samples (FULL CONDITIONING)...{Colors.RESET}")
    with torch.inference_mode():
        # Cloned (Using both boosted embedding and boosted Mel features)
        cloned_wav, _ = engine.audio_decoder.token2wav(
            test_tokens, uuid="test_cloned",
            prompt_token=engine.ref_prompt_token,
            prompt_feat=engine.ref_prompt_feat,
            embedding=engine.ref_prompt_embedding,
            finalize=True
        )
        
        # Neutral (Baseline)
        neutral_embedding = torch.zeros_like(engine.ref_prompt_embedding)
        empty_token = torch.zeros(1, 0, dtype=torch.int32, device=engine.device)
        empty_feat = torch.zeros(1, 0, 80, device=engine.device)
        
        neutral_wav, _ = engine.audio_decoder.token2wav(
            test_tokens, uuid="test_neutral",
            prompt_token=empty_token,
            prompt_feat=empty_feat,
            embedding=neutral_embedding,
            finalize=True
        )

    # 3. Verify Identity
    print(f"\n{Colors.YELLOW}Step 3: Calculating Identity Fidelity...{Colors.RESET}")
    
    # Extract embeddings from the outputs
    # Note: engine.speaker_encoder expects [T] or [1, T] at 16kHz
    # token2wav returns 22050Hz. We need to resample.
    cloned_16k = torchaudio.functional.resample(cloned_wav, 22050, 16000)
    neutral_16k = torchaudio.functional.resample(neutral_wav, 22050, 16000)
    
    cloned_emb = engine.speaker_encoder.extract_embedding(cloned_16k[0])
    neutral_emb = engine.speaker_encoder.extract_embedding(neutral_16k[0])
    
    sim_cloned = calculate_similarity(ref_embedding, cloned_emb)
    sim_neutral = calculate_similarity(ref_embedding, neutral_emb)
    
    print("-" * 50)
    print(f"Reference vs. {Colors.GREEN}CLONED{Colors.RESET} Identity:  {sim_cloned:.4f}")
    print(f"Reference vs. {Colors.GRAY}NEUTRAL{Colors.RESET} Identity: {sim_neutral:.4f}")
    print("-" * 50)
    
    diff = sim_cloned - sim_neutral
    if diff > 0.15:
        print(f"{Colors.GREEN}{Colors.BOLD}PASS: Voice identity successfully transferred! (Improvement: {diff:.4f}){Colors.RESET}")
    elif diff > 0.05:
        print(f"{Colors.YELLOW}WEAK PASS: Identity transferred but similarity is low.{Colors.RESET}")
    else:
        print(f"{Colors.RED}FAIL: Cloned output is no more similar to reference than the neutral baseline.{Colors.RESET}")

    # Save results for manual inspection
    out_dir = Path("scratch/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_dir / "cloned_output.wav", cloned_wav.cpu(), 22050)
    torchaudio.save(out_dir / "neutral_output.wav", neutral_wav.cpu(), 22050)
    print(f"\nSamples saved to {out_dir}/ for manual listening.")

if __name__ == "__main__":
    main()
