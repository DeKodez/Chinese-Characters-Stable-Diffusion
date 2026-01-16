# Chinese Character Generation with Stable Diffusion

A project to generate new Chinese characters using Stable Diffusion by creating training datasets from Unicode character ranges.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Scripts

### `char_to_image.py`

Converts Chinese Unicode characters to images. Can generate images for individual characters, lists of characters, or entire Unicode ranges (e.g., U+4E00 to U+9FFF).

**Usage:**
```bash
python char_to_image.py
```

Or use as a module:
```python
from char_to_image import char_to_image, generate_unicode_range_dataset

# Single character
char_to_image("字", "output.png", platform="macos")

# Unicode range
generate_unicode_range_dataset("U+4E00", "U+9FFF", "char_images")
```

### `char_metadata.py`

Downloads and parses character metadata (meanings, pinyin, stroke counts) from Unihan and CC-CEDICT databases. Creates a unified metadata database for Chinese characters.

**Usage:**
```bash
python char_metadata.py
```

This downloads Unihan and CC-CEDICT (if needed), merges them, and saves to `data/char_metadata.json`.

**Note:** You can run this manually, but `build_training_dataset.py` will automatically generate metadata if it doesn't exist.

### `build_training_dataset.py`

Combines character images with metadata to create a training dataset in JSONL format. Automatically generates metadata and images if they don't exist.

**Usage:**
```bash
# Test with small range
python build_training_dataset.py --test

# Build full dataset
python build_training_dataset.py --start U+4E00 --end U+9FFF
```

Creates a JSONL file with entries like:
```json
{
  "image_path": "char_images/U+4E00_一.png",
  "character": "一",
  "codepoint": "U+4E00",
  "caption": "Chinese character 一 (U+4E00), meaning 'one; a, an; alone', pronounced 'yī', stroke count: 1",
  "definition": "one; a, an; alone",
  "pinyin": "yī",
  "stroke_count": "1"
}
```

## Notes

- The `char_to_image.py` script supports macOS and Windows font paths
- The full CJK range (U+4E00 to U+9FFF) contains ~20,000 characters
- Generated data is saved in `char_images/` and `data/` directories
