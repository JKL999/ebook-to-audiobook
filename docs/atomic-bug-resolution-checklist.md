# Atomic Bug Resolution Checklist

## Overview
This document provides an atomic, step-by-step checklist for resolving the three main bugs in the ebook-to-audiobook project. Each item is a small, actionable task that can be completed independently.

## Bug 1: GPTSoVITS Integration Issues

### Phase 1: Analyze Current Integration
- [ ] Review `src/ebook2audio/voices/gpt_sovits_provider.py` implementation
- [ ] Check if GPTSoVITSProvider is properly registered in voice manager
- [ ] Analyze the inference pipeline in `GPT-SoVITS/GPT_SoVITS/inference.py`
- [ ] Document the current integration flow

### Phase 2: Fix Provider Implementation
- [ ] Update import paths in `gpt_sovits_provider.py`
- [ ] Implement proper model loading mechanism
- [ ] Add proper error handling for missing models
- [ ] Create fallback mechanism if model fails

### Phase 3: Test Integration
- [ ] Create minimal test script for GPTSoVITS provider
- [ ] Test with a simple text input
- [ ] Verify audio output quality
- [ ] Test integration with main pipeline

### Phase 4: Connect to Main Pipeline
- [ ] Update `convert_full_book.py` to use GPTSoVITS
- [ ] Ensure proper voice selection mechanism
- [ ] Test full book conversion with LKY voice
- [ ] Verify chapter transitions

## Bug 2: Repository Organization Issues

### Phase 1: Document Current Issues
- [ ] List all files with incorrect path references
- [ ] Identify all duplicate files
- [ ] Document all hardcoded paths
- [ ] Find all broken imports

### Phase 2: Standardize Paths
- [ ] Create path configuration module
- [ ] Update all absolute paths to relative
- [ ] Fix import statements across all files
- [ ] Update configuration files

### Phase 3: Clean Up Structure
- [ ] Remove duplicate files
- [ ] Organize models into proper directories
- [ ] Consolidate training scripts
- [ ] Update documentation references

### Phase 4: Create Development Guidelines
- [ ] Document proper file organization
- [ ] Create path reference guide
- [ ] Update README with structure info
- [ ] Create `.gitignore` for generated files

## Bug 3: Memory Limitations During Training

### Phase 1: Analyze Current Configuration
- [ ] Review `train_s2_enhanced.py` settings
- [ ] Check batch size in S2 config files
- [ ] Analyze memory usage patterns
- [ ] Document current resource requirements

### Phase 2: Optimize Training Configuration
- [ ] Reduce batch size in S2 config
- [ ] Enable gradient accumulation
- [ ] Implement gradient checkpointing
- [ ] Add memory-efficient data loading

### Phase 3: Implement Advanced Optimizations
- [ ] Enable mixed precision training (fp16)
- [ ] Implement dynamic batch sizing
- [ ] Add memory monitoring
- [ ] Create memory profiling script

### Phase 4: Test and Validate
- [ ] Run S2 training with new settings
- [ ] Monitor memory usage during training
- [ ] Validate model quality
- [ ] Document optimal settings

## Quick Wins (Can be done immediately)

### Documentation and Organization
- [ ] Update bug documentation with file references
- [ ] Create environment setup guide
- [ ] Document GPTSoVITS installation steps
- [ ] Create troubleshooting guide

### Code Fixes
- [ ] Fix obvious import errors
- [ ] Add missing `__init__.py` files
- [ ] Update hardcoded paths to use `os.path`
- [ ] Add proper logging to training scripts

### Configuration
- [ ] Create default configuration templates
- [ ] Add configuration validation
- [ ] Document all configuration options
- [ ] Create example configurations

## Priority Order

1. **Immediate** (Do First):
   - Fix import errors in GPTSoVITS provider
   - Reduce batch size for S2 training
   - Document current file structure

2. **High Priority** (Do Next):
   - Implement proper model loading
   - Fix path references
   - Add memory optimizations

3. **Medium Priority** (Do After):
   - Clean up duplicate files
   - Improve error handling
   - Create comprehensive tests

4. **Low Priority** (Do Last):
   - Optimize performance
   - Add advanced features
   - Create detailed documentation

## Success Criteria

### Bug 1 Success:
- GPTSoVITS provider loads without errors
- Can generate audio with LKY voice
- Integrates with main pipeline
- Produces quality audiobook output

### Bug 2 Success:
- All imports work correctly
- No hardcoded paths remain
- Clear file organization
- Easy to navigate codebase

### Bug 3 Success:
- S2 training completes without OOM
- Memory usage stays under 30GB
- Training time is reasonable
- Model quality is maintained

## Next Steps
1. Start with the immediate priority items
2. Test each fix before moving to the next
3. Document all changes made
4. Update this checklist as items are completed