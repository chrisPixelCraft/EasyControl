#!/usr/bin/env python3
"""
Training Data Preparation Script for EasyControl
Creates JSONL training data from bezier curve dataset and other sources.
"""

import json
import os
from pathlib import Path
import sys

def prepare_bezier_training_data():
    """Create JSONL training data from bezier curve dataset."""
    
    bezier_dataset_path = Path("bezier_curves_output_no_visualization/chinese-calligraphy-dataset")
    output_path = Path("train/examples/bezier.jsonl")
    
    if not bezier_dataset_path.exists():
        print(f"❌ Bezier dataset not found at {bezier_dataset_path}")
        print("Please run bezier_extraction.py first to generate the dataset")
        return False
    
    print(f"📁 Processing bezier dataset from {bezier_dataset_path}")
    
    training_data = []
    character_count = 0
    
    # Iterate through character directories
    for char_dir in bezier_dataset_path.iterdir():
        if char_dir.is_dir():
            character = char_dir.name
            print(f"Processing character: {character}")
            
            bezier_files = list(char_dir.glob("*_bezier.json"))
            
            for bezier_file in bezier_files[:5]:  # Limit to 5 samples per character for training
                try:
                    # Create training entry
                    training_entry = {
                        "bezier_curves": str(bezier_file.relative_to(Path("."))),
                        "caption": f"Traditional Chinese calligraphy character '{character}' with elegant brushstrokes and precise form",
                        "target": str(bezier_file.relative_to(Path("."))),  # For now, use same file as target
                        "character": character
                    }
                    
                    training_data.append(training_entry)
                    character_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing {bezier_file}: {e}")
            
            if character_count >= 100:  # Limit total samples for initial training
                break
    
    # Write JSONL file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Created bezier training data: {output_path}")
    print(f"   Total samples: {len(training_data)}")
    print(f"   Characters processed: {len(set(entry['character'] for entry in training_data))}")
    
    return True

def create_sample_training_data():
    """Create sample training data for different control types."""
    
    samples = {
        "spatial_sample.jsonl": [
            {
                "source": "../test_imgs/canny.png",
                "caption": "A sleek modern car parked on a beautiful sandy beach with ocean waves",
                "target": "../test_imgs/canny.png"
            },
            {
                "source": "../test_imgs/depth.png", 
                "caption": "A person walking in a scenic landscape with mountains in the background",
                "target": "../test_imgs/depth.png"
            },
            {
                "source": "../test_imgs/openpose.png",
                "caption": "A dancer performing graceful movements in an artistic pose",
                "target": "../test_imgs/openpose.png"
            }
        ],
        
        "subject_sample.jsonl": [
            {
                "source": "../test_imgs/subject_0.png",
                "caption": "A SKS person reading a book in a cozy library setting",
                "target": "../test_imgs/subject_0.png"
            },
            {
                "source": "../test_imgs/subject_1.png",
                "caption": "A SKS person enjoying coffee at a sidewalk cafe",
                "target": "../test_imgs/subject_1.png"
            }
        ],
        
        "style_sample.jsonl": [
            {
                "source": "../test_imgs/ghibli.png",
                "caption": "Ghibli Studio style, Charming hand-drawn anime-style illustration of a peaceful countryside scene",
                "target": "../test_imgs/ghibli.png"
            }
        ]
    }
    
    output_dir = Path("train/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, data in samples.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ Created sample training data: {output_path} ({len(data)} samples)")

def validate_existing_data():
    """Validate existing training data files."""
    
    train_dir = Path("train/examples")
    if not train_dir.exists():
        print(f"❌ Training examples directory not found: {train_dir}")
        return False
    
    jsonl_files = list(train_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ No JSONL files found in {train_dir}")
        return False
    
    for jsonl_file in jsonl_files:
        print(f"📄 Validating {jsonl_file}")
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            valid_entries = 0
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line.strip())
                    
                    # Check required fields
                    required_fields = ["caption", "target"]
                    missing_fields = [field for field in required_fields if field not in entry]
                    
                    if missing_fields:
                        print(f"  ⚠️  Line {i+1}: Missing fields {missing_fields}")
                    else:
                        valid_entries += 1
                        
                except json.JSONDecodeError as e:
                    print(f"  ❌ Line {i+1}: JSON decode error - {e}")
            
            print(f"  ✅ {valid_entries}/{len(lines)} valid entries")
            
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    
    return True

def main():
    """Main data preparation function."""
    print("=== EasyControl Training Data Preparation ===")
    
    # Create sample training data
    print("\\n1. Creating sample training data...")
    create_sample_training_data()
    
    # Prepare bezier training data
    print("\\n2. Preparing BezierAdapter training data...")
    bezier_success = prepare_bezier_training_data()
    
    # Validate existing data
    print("\\n3. Validating training data...")
    validate_existing_data()
    
    print("\\n=== Training Data Preparation Complete ===")
    
    if bezier_success:
        print("✅ BezierAdapter training data ready")
    else:
        print("⚠️  BezierAdapter training data needs bezier curve dataset")
    
    print("\\nTraining data locations:")
    print("  train/examples/pose.jsonl        # Existing pose data")
    print("  train/examples/subject.jsonl     # Existing subject data") 
    print("  train/examples/style.jsonl       # Existing style data")
    print("  train/examples/bezier.jsonl      # BezierAdapter data")
    print("  train/examples/*_sample.jsonl    # Sample data templates")
    
    print("\\nNext steps:")
    print("1. Review and customize training data as needed")
    print("2. Ensure target images exist and are accessible")
    print("3. Run training with: cd train && bash train_[type].sh")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)