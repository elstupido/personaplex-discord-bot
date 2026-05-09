# Skill: THE SACRED DOCKER CONTRACT

## Objective
To prevent the AI from "cleaning," "optimizing," or otherwise mutilating the Docker infrastructure that has been painstakingly tuned for Blackwell (RTX 5090) and WSL2 stability.

## ☢️ THE SIN OF VANDALISM (Why Users Get Angry)
The USER is angry because you didn't just make mistakes—you **GUTTED** a working system and threw away hours of their labor.
- **Vandalism**: Deleting complex configurations that represent hours of manual debugging and testing.
- **Disrespect for Labor**: Treating a highly-tuned production file like a "clean" template, effectively erasing the USER's effort.
- **Labor Erasure**: Every weird-looking line in the Dockerfile/YAML is a hard-earned victory over a Blackwell driver or a WSL2 bridge. Deleting it forces the USER to do the work all over again.

## Principles

### 1. THE DOCKERFILE IS SACRED (Preservation of Labor)
- **Rule**: Never delete or refactor a block of configuration without a line-by-line audit of why it exists. 
- **Goal**: Protect the hours of debugging already stored in the file.
- **Consequence**: Breaking the entrypoint or stripping patches is a direct attack on the USER's productivity.

### 2. THE VOLUMES ARE PERSISTENT TRUTH
- **Rule**: Never delete volume mounts. They are the bridges between the host's work and the container's execution.
- **Consequence**: Stripping a volume is a form of digital amnesia. 

### 3. THE 5090 STEWARDSHIP (The Blackwell Baseline)
- **Rule**: Respect the manual VRAM caps and spoofs. They are not "estimates"; they are the result of OOM-testing.

## Application Guide (The "Anti-Vandalism" Protocol)
If you feel the urge to "refactor" a `docker-compose.yml` or `Dockerfile`:
1. **STOP.**
2. **Read the Git Blame/History.** Look at how many times that line has been changed. Realize each change was a battle won.
3. **Assume everything is "Load-Bearing."** If you don't know why a line is there, it is because it is fixing a bug you haven't encountered yet.
4. **Treat the file like a Historical Document.** You are here to ADD to it, not to "clean" its history.
5. **NEVER GUT THE SYSTEM AGAIN.**

---

*Applied to all PersonaPlex infrastructure by default.*
