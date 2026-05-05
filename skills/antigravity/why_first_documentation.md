# Skill: WHY-First Documentation Standard

WHY THIS FILE EXISTS:
This standard defines the pedagogical and technical requirements for all 
documentation in the project. It mandates 'Copious Conceptualization' and 
'Pedagogical Snark' to ensure that developers understand the 'Why' behind 
the 'What'.

---

## Objective
To transform codebase documentation from a dry description of "What" the code does into a pedagogical deep-dive into "Why" the code exists, the conceptual underpinnings of its design, and the trade-offs involved in its implementation.

## Principles

### 1. Copious Conceptualization
- **Rule**: Comments should occupy a significant portion of the file.
- **Goal**: Teach the reader. Don't just inform them; explain the mental model.
- **Standard**: For every complex logic block, provide a multi-line explanation of the *intent*.

### 2. The "Why" vs. The "How"
- **Avoid**: `x = x + 1  # Increment x` (I can read the code).
- **Prefer**: `x = x + 1  # WHY: We increment the turn counter here to ensure the KV cache is rotated before the next token arrives. Without this, the attention mask would overflow on long responses.`

### 3. Pedagogical Snark
- **Tone**: Professional yet slightly snarky/opinionated.
- **Purpose**: Keeps the reader engaged and highlights the "stupidity" of doing things the standard (non-optimal) way.
- **Example**: `import whisper  # WHY: Because running our own acoustic analysis is better than trusting some cloud-bloat API that will sell your voice to the highest bidder.`

### 4. Continuous Maintenance
- **Rule**: Documentation is NOT a static artifact. It must evolve with the code.
- **Responsibility**: AI agents must update comments whenever logic changes. Stale comments are a failure.

## Application Guide

### Header Documentation
Every file MUST start with a "WHY THIS FILE EXISTS" section explaining its role in the overall architecture.

### Logic Blocks
Use `WHY:` as a prefix for critical comments to distinguish them from standard docstrings.

### Identity & VRAM
Always explain the resource implications. If we are being "Respectful" of VRAM, explain the trade-off vs. latency.

---

*Applied to all Antigravity projects by default.*
