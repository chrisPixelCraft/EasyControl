# EasyControl Task Completion Checklist

## Before Starting Work
- [ ] Activate the correct conda environment (`easycontrol`)
- [ ] Verify CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Check GPU memory: `nvidia-smi`
- [ ] Ensure all dependencies are installed: `pip install -r requirements.txt`

## Code Development
- [ ] Follow PEP 8 style guidelines
- [ ] Add type hints to function parameters and return values
- [ ] Include docstrings for new functions and classes
- [ ] Use descriptive variable and function names
- [ ] Implement proper error handling

## Model Training Tasks
- [ ] Prepare training data in correct JSONL format
- [ ] Set appropriate `cond_size` (384-512 or higher)
- [ ] Configure `noise_size` (1024 recommended)
- [ ] Set LoRA rank to 128
- [ ] Use learning rate of 1e-4
- [ ] Set batch size to 1 (required for multi-resolution)
- [ ] Configure validation steps and checkpointing
- [ ] Clear cache after each training iteration

## Inference Tasks
- [ ] Load the correct base model (`black-forest-labs/FLUX.1-dev`)
- [ ] Set appropriate LoRA weights (typically [1])
- [ ] Configure `cond_size` to match training
- [ ] Set guidance scale (default 3.5)
- [ ] Use 25 inference steps for quality/speed balance
- [ ] Call `clear_cache(pipe.transformer)` after generation
- [ ] Save outputs with descriptive filenames

## Multi-Condition Setup
- [ ] Ensure subject LoRA path comes before spatial LoRA path
- [ ] Configure `lora_weights` correctly for each condition
- [ ] Verify input image formats (RGB, correct dimensions)
- [ ] Test combinations before deployment

## Testing & Validation
- [ ] Test single-condition generation
- [ ] Test multi-condition generation
- [ ] Verify output quality and consistency
- [ ] Check memory usage during processing
- [ ] Test with different image sizes and aspect ratios

## Documentation
- [ ] Update README if adding new features
- [ ] Document new parameters and their usage
- [ ] Include usage examples
- [ ] Update configuration files if needed

## Version Control
- [ ] Stage and commit changes with meaningful messages
- [ ] Push to appropriate branch
- [ ] Create pull request if needed
- [ ] Update version tags for releases

## Final Verification
- [ ] Run all inference scripts successfully
- [ ] Verify model outputs meet quality standards
- [ ] Check for memory leaks or performance issues
- [ ] Ensure all temporary files are cleaned up
- [ ] Confirm all requirements are documented