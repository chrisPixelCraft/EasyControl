# EasyControl Development Guidelines

## Code Style & Conventions

### Python Style
- **PEP 8**: Follow standard Python style guidelines
- **Type Hints**: Use type hints for function parameters and return values
- **Docstrings**: Document functions and classes with clear descriptions
- **Import Organization**: Group imports (standard, third-party, local)

### Variable Naming
- **snake_case**: For functions and variables
- **PascalCase**: For classes
- **UPPERCASE**: For constants
- **Descriptive Names**: Use clear, meaningful names

### Function Design
- **Single Responsibility**: Each function should do one thing well
- **Clear Parameters**: Use descriptive parameter names
- **Error Handling**: Include proper exception handling
- **Cache Management**: Always call `clear_cache()` after generation

## Model Architecture Patterns

### LoRA Integration
- Use `set_single_lora()` for single condition
- Use `set_multi_lora()` for multiple conditions
- Subject LoRA path should come before spatial LoRA path in multi-condition
- Always specify `cond_size` parameter (typically 512)

### Memory Management
- Clear KV cache after each generation: `clear_cache(pipe.transformer)`
- Use `torch.bfloat16` for memory efficiency
- Batch size limited to 1 for training due to multi-resolution images

### Configuration Standards
- **Guidance Scale**: Default 3.5, adjust based on results
- **Inference Steps**: Default 25 for quality/speed balance
- **Max Sequence Length**: 512 for prompts
- **LoRA Rank**: 128 (recommended for training)
- **Learning Rate**: 1e-4 (recommended for training)

## File Organization
- Keep training and inference code separated
- Use consistent naming for model files (.safetensors)
- Store test images in `test_imgs/` directory
- Use descriptive filenames for outputs

## Testing & Validation
- Test with different control types before deployment
- Validate multi-condition combinations
- Check memory usage during training
- Test inference with various image sizes

## Performance Guidelines
- **GPU Memory**: Monitor usage, especially during training
- **Batch Processing**: Process images individually to avoid memory issues
- **Model Loading**: Load models once and reuse
- **Caching**: Utilize KV cache for efficiency

## Documentation Standards
- Update README when adding new features
- Include usage examples in docstrings
- Document parameter requirements
- Provide troubleshooting guides

## Version Control
- Use meaningful commit messages
- Tag releases with version numbers
- Document breaking changes
- Keep training scripts and configs in version control