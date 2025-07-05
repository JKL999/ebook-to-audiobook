# Ebook-to-Audiobook Bug Documentation

## Project Overview
A program that converts ebooks (PDF) to audiobooks (MP3) using trained voice models. The first implementation targets Lee Kuan Yew's works using his trained voice model.

## Current Bugs

### Bug 1: GPTSoVITS Integration Issues
**Status:** Open  
**Priority:** Critical  
**Description:** Integration and use of the GPTSoVITS technology and process to generate a proper model and integrate into the audiobook generation pipeline.

#### Problem Details
- Voice model generation process not properly integrated
- Pipeline connectivity issues between GPTSoVITS and main audiobook generation
- Model output format compatibility issues

#### Related Files
- TBD (will be documented after file structure analysis)

#### Reproduction Steps
1. TBD

#### Proposed Solutions
- TBD

---

### Bug 2: Repository Organization Issues
**Status:** Open  
**Priority:** High  
**Description:** Repository has been worked on in multiple places with multiple code LLM agents from various computers, resulting in scattered and unorganized file directory references and files.

#### Problem Details
- Inconsistent file paths across different development environments
- Duplicate files in multiple locations
- Missing or broken references to dependencies
- Unclear project structure

#### Related Files
- TBD (will be documented after file structure analysis)

#### Impact
- Development efficiency reduced
- Risk of using outdated files
- Difficulty in maintaining and debugging

#### Proposed Solutions
- Create comprehensive file structure documentation
- Consolidate duplicate files
- Establish clear directory conventions
- Update all path references

---

### Bug 3: Memory Limitations During Training
**Status:** Open  
**Priority:** High  
**Description:** Second stage model training crashes due to memory limitations on I7 3070GPU with 32GB RAM Linux machine.

#### Problem Details
- OOM (Out of Memory) errors during second stage training
- Process crashes despite 32GB RAM and 3070 GPU

#### System Specifications
- CPU: Intel i7
- GPU: NVIDIA RTX 3070
- RAM: 32GB
- OS: Linux

#### Related Files
- TBD (will be documented after file structure analysis)

#### Attempted Solutions
- TBD

#### Proposed Solutions
- Reduce batch size
- Implement gradient checkpointing
- Use mixed precision training
- Optimize data loading pipeline
- Consider model pruning or quantization

---

## Action Items
1. Complete file structure analysis
2. Document all related files for each bug
3. Create reproduction steps
4. Develop solution strategies
5. Implement fixes incrementally