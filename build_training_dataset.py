"""
Build a training dataset by combining character images with metadata.
Creates a JSONL file suitable for Stable Diffusion fine-tuning.
"""

import json
from pathlib import Path
from char_to_image import generate_unicode_range_dataset
from char_metadata import (
    load_metadata_db,
    get_character_metadata,
    generate_caption,
)


def build_training_dataset(
    start: str = "U+4E00",
    end: str = "U+9FFF",
    images_dir: str = "char_images",
    metadata_path: str = "data/char_metadata.json",
    output_jsonl: str = "training_dataset.jsonl",
    regenerate_images: bool = False,
    regenerate_metadata: bool = False
):
    """
    Build a complete training dataset with images and captions.
    
    Args:
        start: Start Unicode range
        end: End Unicode range
        images_dir: Directory containing character images
        metadata_path: Path to metadata JSON file
        output_jsonl: Output JSONL file for training
        regenerate_images: Whether to regenerate images if they exist
        regenerate_metadata: Whether to regenerate metadata if it exists
    """
    print("=" * 60)
    print("Building Training Dataset for Chinese Character Generation")
    print("=" * 60)
    
    # Step 1: Ensure metadata exists
    metadata_file = Path(metadata_path)
    if not metadata_file.exists() or regenerate_metadata:
        print("\n[1/4] Generating metadata database...")
        print("This will download Unihan and CC-CEDICT if needed.")
        from char_metadata import (
            download_unihan_data,
            download_cedict,
            load_unihan_database,
            parse_cedict,
            merge_character_metadata,
            save_metadata_db
        )
        
        try:
            download_unihan_data()
        except Exception as e:
            print(f"Warning: Could not download Unihan: {e}")
        
        unihan_data = load_unihan_database()
        
        cedict_data = {}
        try:
            download_cedict()
            cedict_path = Path("data/cedict.txt")
            if cedict_path.exists():
                cedict_data = parse_cedict(cedict_path)
        except Exception as e:
            print(f"Warning: Could not download CC-CEDICT: {e}")
        
        if unihan_data or cedict_data:
            merged = merge_character_metadata(unihan_data, cedict_data)
            save_metadata_db(merged, metadata_path)
            metadata_db = merged
        else:
            print("No metadata sources available. Creating empty metadata database.")
            metadata_db = {}
    else:
        print("\n[1/4] Loading existing metadata database...")
        metadata_db = load_metadata_db(metadata_path)
    
    # Step 2: Generate images if needed
    images_path = Path(images_dir)
    if not images_path.exists() or not any(images_path.glob("*.png")) or regenerate_images:
        print(f"\n[2/4] Generating character images ({start} to {end})...")
        print("This may take a while for large ranges...")
        generate_unicode_range_dataset(
            start, end,
            output_dir=images_dir,
            progress_interval=500
        )
    else:
        print(f"\n[2/4] Using existing images in {images_dir}/")
    
    # Step 3: Build training dataset
    print("\n[3/4] Building training dataset...")
    training_data = []
    images_path = Path(images_dir)
    
    # Parse Unicode range
    from char_to_image import parse_unicode_range
    start_code, end_code = parse_unicode_range(start, end)
    
    generated_count = 0
    missing_metadata_count = 0
    missing_image_count = 0
    
    for code_point in range(start_code, end_code + 1):
        try:
            char = chr(code_point)
            unicode_code = f"U+{code_point:04X}"
            
            # Find corresponding image
            # Try different filename patterns
            image_patterns = [
                f"{unicode_code}_{char}.png",
                f"{unicode_code}.png",
                f"{char}.png"
            ]
            
            image_path = None
            for pattern in image_patterns:
                potential_path = images_path / pattern
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if not image_path:
                missing_image_count += 1
                continue
            
            # Get metadata
            metadata = get_character_metadata(char, metadata_db)
            
            # Generate caption
            caption = generate_caption(char, metadata)
            
            # Create training entry
            entry = {
                "image_path": str(image_path.relative_to(Path.cwd())),
                "character": char,
                "codepoint": unicode_code,
                "caption": caption
            }
            
            # Add metadata fields if available
            if metadata:
                entry.update({
                    "definition": metadata.get("definition_en", ""),
                    "pinyin": metadata.get("mandarin_pinyin", ""),
                    "stroke_count": metadata.get("stroke_count", ""),
                })
            else:
                missing_metadata_count += 1
            
            training_data.append(entry)
            generated_count += 1
            
            if generated_count % 1000 == 0:
                print(f"  Processed {generated_count} entries...")
        
        except (ValueError, OverflowError):
            continue
    
    # Step 4: Save JSONL file
    print(f"\n[4/4] Saving training dataset to {output_jsonl}...")
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print(f"  Total entries: {len(training_data)}")
    print(f"  Images found: {generated_count}")
    print(f"  Missing images: {missing_image_count}")
    print(f"  Missing metadata: {missing_metadata_count}")
    print(f"  Output file: {output_jsonl}")
    print("=" * 60)
    
    # Show sample entries
    if training_data:
        print("\nSample entries:")
        for i, entry in enumerate(training_data[:3]):
            print(f"\n{i+1}. Character: {entry['character']}")
            print(f"   Caption: {entry['caption']}")
    
    return training_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build training dataset for Chinese character generation")
    parser.add_argument("--start", type=str, default="U+4E00", help="Start Unicode range")
    parser.add_argument("--end", type=str, default="U+9FFF", help="End Unicode range")
    parser.add_argument("--images-dir", type=str, default="char_images", help="Directory with character images")
    parser.add_argument("--metadata", type=str, default="data/char_metadata.json", help="Metadata JSON file")
    parser.add_argument("--output", type=str, default="training_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--regenerate-images", action="store_true", help="Regenerate images even if they exist")
    parser.add_argument("--regenerate-metadata", action="store_true", help="Regenerate metadata even if it exists")
    parser.add_argument("--test", action="store_true", help="Test with small range (U+4E00 to U+4E0F)")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with small range...")
        build_training_dataset(
            start="U+4E00",
            end="U+4E0F",
            images_dir="test_char_images",
            metadata_path=args.metadata,
            output_jsonl="test_training_dataset.jsonl",
            regenerate_images=args.regenerate_images,
            regenerate_metadata=args.regenerate_metadata
        )
    else:
        build_training_dataset(
            start=args.start,
            end=args.end,
            images_dir=args.images_dir,
            metadata_path=args.metadata,
            output_jsonl=args.output,
            regenerate_images=args.regenerate_images,
            regenerate_metadata=args.regenerate_metadata
        )
