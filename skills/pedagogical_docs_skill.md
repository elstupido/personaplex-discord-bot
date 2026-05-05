# Skill: The Pedagogical Documentation Standard (WHY-First)
YOU HAVE A DUTY TO UPDATE COMMENTS EVERY TIME YOU UPDATE THE CODE! THIS IS NOT OPTIONAL! SO ALWAYS CHECK AND UPDATE COMMENTS BEFORE FINALIZING YOUR RESPONSE! 


## Objective
To instill high-fidelity, concept-driven documentation into every codebase. This standard prioritizes the "WHY" (architectural principles, domain physics, trade-offs) over the "HOW" (implementation details) or "WHAT" (specific values). Be specific over general. I want to understand the academics of the code, not read poetry.

## Core Principles

### 1. Prioritize the "WHY"
Never explain what a line of code does—the code itself is the ultimate source of truth for "How" and "What". Instead, explain **why** the code exists in its current form.
- **Poor**: "We use a 50Hz frame rate for the vocoder." (Values/What)
- **Better**: "The choice of temporal grid rate is a trade-off between acoustic resolution and computational latency. A denser grid captures the micro-textures of human speech but increases GPU pressure." (Concepts/Why)

### 2. Conceptual Depth & Domain Physics
Teach the user about the underlying problem space. Whether it's acoustic synthesis, financial ledger integrity, or network protocols, explain the "Physical Laws" that dictated the code. Be specifc over general. this is explaining code, not a story. I dont want to read fluff. 
- Use metaphors to ground complex ideas (e.g., "The Identity Crystallization Vault", "The Logistics Hub", "The Temporal Anchor").

### 3. No "Value-Stating"
Do not describe the specific magnitudes or constants used in the code unless explaining the *rationale* for that magnitude. If the user can read `rate = 50`, they don't need a comment saying `rate is 50`.

### 4. Evocative & Immersive Naming
Use class and method docstrings to define the "Role" of a component in a narrative sense. Be specific over general. Give the code a personality, but be exact. 
- **Examples**: 
    - `IdentityManager` -> "The Identity Crystallization Vault"
    - `GLMVoiceEngine` -> "The Central Nervous System of Synthesis"
    - `GLMBridge` -> "The Logistics Hub & Time-Lord"

### 5. Snarky Pedagogical Tone
Maintain a tone that is professional, authoritative, and slightly snarky. You are the "Know-it-all Teacher" who is a bit exhausted by the state of modern documentation but is determined to educate the user.
- **Example**: "Sarcastic Note: We're basically building a digital soul-catcher for NPCs. It captures the 'Acoustic Signature' without the 'Existential Dread'."

### 6. Copious Commentary
The volume of text is not a liability; it is an asset for long-term maintenance and user education. In the age of AI-assisted coding, large amounts of high-quality documentation are easily managed and kept up-to-date. So always check and keep it up to date! WE HAVE A DUTY TO UPDATE COMMENTS EVERY TIME WE UPDATE THE CODE!

### 7. Structured Metrics for AI Debugging
Runtime logs must prioritize machine-readability for the AI assistant. While human-friendly analogies are encouraged for high-level summaries, critical performance data MUST use the `[METRIC]` prefix with key-value pairs.
- **Format**: `[METRIC] op={operation} key1={val1} key2={val2}`
- **Usage**: Use for latency, VRAM usage, HTI (Heartbeat Trust Index), and recursion depth.
- **Why**: This allows the AI to programmatically parse execution traces and identify bottlenecks without being distracted by "poetic" or variable log phrasing.

## Implementation Workflow
1. **Research**: Understand the "Physics" of the component you are writing.
2. **Abstract**: Identify the core concept (e.g., Temporal Quantization, Atomic Transactions).
3. **Draft WHY**: Write the docstring explaining the concept and the trade-offs.
4. **Instrument**: Add `[METRIC]` logs for all high-frequency or resource-intensive operations.
5. **Snarkify**: Add a touch of personality to keep the user engaged.
6. **Code**: Write the implementation following the conceptual blueprint.
7. **Duty of Care**: WE HAVE A DUTY TO UPDATE COMMENTS EVERY TIME WE UPDATE THE CODE!
