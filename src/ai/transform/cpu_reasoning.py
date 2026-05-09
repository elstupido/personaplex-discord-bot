# WHY THIS FILE EXISTS:
# This is the 'Body-Brain' reasoning core. 
# It runs a lightweight 1.5B model on the CPU to protect the GPU VRAM for Fish Audio.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger
from typing import AsyncGenerator

@StupidRegistry.register("cpu_reasoning")
class CPUReasoningExpert(StupidStep):
    """
    The 'Thinker' that lives in the System RAM. 🧠💻
    """
    def __init__(self, name: str):
        super().__init__(name)
        # WHY: Qwen 2.5 1.5B is the sweet spot for CPU-bound reasoning.
        # It's small enough to stay in System RAM without choking the OS,
        # but smart enough to handle Discord banter.
        self.model_id = os.getenv("REASONING_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.tokenizer = None
        self.model = None
        logger.info(f"✨ [CPUReasoning] '{name}' ({self.model_id}) initialized.")

    async def warmup(self):
        """
        Pre-heat the 'Thinker'. 🦾
        
        WHY: Loading a 1.5B model into RAM is a heavy synchronous operation. 
        We move it to a thread to avoid blocking the Discord heartbeats.
        """
        if self.model is None:
            import asyncio
            await asyncio.to_thread(self._ensure_model)

    def _ensure_model(self):
        if self.model is None:
            logger.info(f"📥 [CPUReasoning] Loading {self.model_id} to System RAM...")
            
            # WHY: Unlocking the CPU's full potential. 🏎️
            threads = os.cpu_count() or 4
            torch.set_num_threads(threads)
            torch.set_flush_denormal(True) # Prevent 'Denormal' slowdowns
            logger.info(f"🏎️ [CPUReasoning] Parallelism enabled: {threads} threads.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # WHY: We use float32 on CPU. float16 is NOT natively supported by standard 
            # CPU ALUs and causes the 'probability tensor contains nan' crashes.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            logger.info(f"✅ [CPUReasoning] {self.model_id} ready on CPU.")

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        await self.warmup()
        
        user_input = data.content
        if data.type != "text" or user_input is None:
            yield data
            return

        # WHY: Empty prompts can cause the probability tensor to collapse into NaNs.
        if not user_input.strip():
            logger.warning("⚠️ [Think] Empty prompt received. Bypassing reasoning.")
            data.content = "I'm sorry, I didn't hear anything. Could you repeat that?"
            yield data
            return

        # Qwen 2.5 Chat Template
        messages = [
            {"role": "user", "content": user_input}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")

        # WHY: Generation is a heavy synchronous operation. 
        # We MUST move it to a thread to avoid stalling the Discord heartbeat 
        # and causing the voice gateway to disconnect.
        import asyncio
        def _generate():
            with torch.no_grad():
                return self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.1
                )
        
        outputs = await asyncio.to_thread(_generate)
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        logger.info(f"💡 [CPUReasoning] Response: \"{response}\"")
        
        data.content = response
        data.type = "text"
        yield data
