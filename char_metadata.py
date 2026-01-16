"""
Fetch and manage Chinese character metadata (meanings, readings, etc.)
for characters in the CJK Unified Ideographs range.

This script downloads the Unihan and CC-CEDICT databases and parses them.
We use both databases as CC-CEDICT is probably better but fails for rare characters.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Union
import urllib.request
from collections import defaultdict


def download_unihan_data(output_dir: str = "data") -> Path:
    """
    Download Unihan database from Unicode Consortium.
    
    Args:
        output_dir: Directory to save downloaded files
    
    Returns:
        Path to downloaded Unihan zip file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    unihan_url = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
    zip_path = output_path / "Unihan.zip"
    
    if zip_path.exists():
        print(f"Unihan data already exists at {zip_path}")
        return zip_path
    
    print(f"Downloading Unihan database from {unihan_url}...")
    print("This may take a few minutes (~10MB download)...")
    
    try:
        urllib.request.urlretrieve(unihan_url, zip_path)
        print(f"Downloaded to {zip_path}")
        return zip_path
    except Exception as e:
        print(f"Error downloading Unihan data: {e}")
        print("You can manually download from: https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip")
        raise


def parse_unihan_file(unihan_txt_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse Unihan database text file.
    
    Args:
        unihan_txt_path: Path to Unihan text file (e.g., Unihan_Readings.txt)
    
    Returns:
        Dictionary mapping character to metadata fields
    """
    char_data: Dict[str, Dict[str, str]] = defaultdict(dict)
    
    if not unihan_txt_path.exists():
        print(f"Warning: {unihan_txt_path} not found")
        return char_data
    
    print(f"Parsing {unihan_txt_path}...")
    
    with open(unihan_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Format: U+4E00	kDefinition	one; a, an; alone
            parts = line.split('\t')
            if len(parts) >= 3:
                codepoint = parts[0]  # e.g., "U+4E00"
                field = parts[1]      # e.g., "kDefinition"
                value = parts[2]      # e.g., "one; a, an; alone"
                
                # Convert codepoint to character
                try:
                    code_int = int(codepoint[2:], 16)  # Remove "U+" and parse hex
                    char = chr(code_int)
                    char_data[char][field] = value
                except (ValueError, OverflowError):
                    continue
    
    print(f"Parsed {len(char_data)} characters from {unihan_txt_path.name}")
    return char_data


def load_unihan_database(data_dir: str = "data") -> Dict[str, Dict[str, str]]:
    """
    Load complete Unihan database.
    
    Args:
        data_dir: Directory containing Unihan files
    
    Returns:
        Dictionary mapping character to all metadata
    """
    data_path = Path(data_dir)
    unihan_dir = data_path / "Unihan"
    
    # If Unihan.zip exists but not extracted, extract it
    zip_path = data_path / "Unihan.zip"
    if zip_path.exists():
        # Check if files are already extracted directly in data/ or in Unihan/ subdirectory
        has_unihan_dir = unihan_dir.exists()
        has_files_directly = any(data_path.glob("Unihan_*.txt"))
        
        if not has_unihan_dir and not has_files_directly:
            import zipfile
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
        elif has_files_directly and not has_unihan_dir:
            # Files are in data/ directly, create Unihan/ and move them
            print("Organizing Unihan files into Unihan/ subdirectory...")
            unihan_dir.mkdir(exist_ok=True)
            for unihan_file in data_path.glob("Unihan_*.txt"):
                unihan_file.rename(unihan_dir / unihan_file.name)
    
    # Check for Unihan files in either location
    if unihan_dir.exists():
        search_dir = unihan_dir
    elif any(data_path.glob("Unihan_*.txt")):
        search_dir = data_path
    else:
        print("Unihan directory not found. Please download Unihan data first.")
        return {}
    
    # Parse key Unihan files
    char_data: Dict[str, Dict[str, str]] = defaultdict(dict)
    
    # Most important files for our use case
    files_to_parse = [
        "Unihan_Readings.txt",      # kMandarin, kCantonese, etc.
        "Unihan_DictionaryLikeData.txt",  # kDefinition
        "Unihan_IRGSources.txt",     # kRSUnicode (radical info)
        "Unihan_OtherMappings.txt",  # kTotalStrokes
    ]
    
    for filename in files_to_parse:
        file_path = search_dir / filename
        if file_path.exists():
            file_data = parse_unihan_file(file_path)
            # Merge into main dictionary
            for char, fields in file_data.items():
                char_data[char].update(fields)
        else:
            print(f"Warning: {filename} not found in Unihan directory")
    
    print(f"Loaded metadata for {len(char_data)} characters from Unihan")
    return char_data


def download_cedict(output_dir: str = "data") -> Path:
    """
    Download CC-CEDICT dictionary.
    
    Args:
        output_dir: Directory to save downloaded file
    
    Returns:
        Path to downloaded CC-CEDICT file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # CC-CEDICT download URL (latest version)
    cedict_url = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt"
    cedict_path = output_path / "cedict.txt"
    
    if cedict_path.exists():
        print(f"CC-CEDICT already exists at {cedict_path}")
        return cedict_path
    
    print(f"Downloading CC-CEDICT from {cedict_url}...")
    print("This may take a few minutes (~5MB download)...")
    
    try:
        urllib.request.urlretrieve(cedict_url, cedict_path)
        print(f"Downloaded to {cedict_path}")
        return cedict_path
    except Exception as e:
        print(f"Error downloading CC-CEDICT: {e}")
        print("You can manually download from: https://www.mdbg.net/chinese/dictionary?page=cedict")
        raise


def parse_cedict(cedict_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse CC-CEDICT dictionary file.
    
    Format: 一 一 [yi1] /one; a, an; alone/
    
    Args:
        cedict_path: Path to CC-CEDICT file
    
    Returns:
        Dictionary mapping character to CEDICT data
    """
    char_data: Dict[str, Dict[str, str]] = {}
    
    if not cedict_path.exists():
        print(f"Warning: {cedict_path} not found")
        return char_data
    
    print(f"Parsing CC-CEDICT from {cedict_path}...")
    
    with open(cedict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Format: 一 一 [yi1] /one; a, an; alone/
            # Extract: traditional, simplified, pinyin, definition
            match = re.match(r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/([^/]+)/', line)
            if match:
                traditional, simplified, pinyin, definition = match.groups()
                
                # Store for both traditional and simplified if different
                for char in [traditional, simplified]:
                    if len(char) == 1:  # Single character only
                        if char not in char_data:
                            char_data[char] = {
                                'cedict_pinyin': pinyin,
                                'cedict_definition': definition,
                                'cedict_traditional': traditional,
                                'cedict_simplified': simplified
                            }
    
    print(f"Parsed {len(char_data)} single-character entries from CC-CEDICT")
    return char_data


def merge_character_metadata(
    unihan_data: Dict[str, Dict[str, str]],
    cedict_data: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """
    Merge Unihan and CC-CEDICT data, preferring richer definitions.
    
    Args:
        unihan_data: Unihan character metadata
        cedict_data: CC-CEDICT character metadata
    
    Returns:
        Merged metadata dictionary
    """
    merged = {}
    all_chars = set(unihan_data.keys()) | set(cedict_data.keys())
    
    for char in all_chars:
        entry = {}
        
        # Start with Unihan data
        if char in unihan_data:
            entry.update(unihan_data[char])
        
        # Add/override with CC-CEDICT if available
        if char in cedict_data:
            cedict_entry = cedict_data[char]
            # Prefer CC-CEDICT definition if it's more detailed
            if 'cedict_definition' in cedict_entry:
                if 'kDefinition' not in entry or len(cedict_entry['cedict_definition']) > len(entry.get('kDefinition', '')):
                    entry['definition'] = cedict_entry['cedict_definition']
                else:
                    entry['definition'] = entry.get('kDefinition', '')
            
            # Add CEDICT-specific fields
            entry.update({k: v for k, v in cedict_entry.items() if k.startswith('cedict_')})
        
        # Standardize field names
        standardized: Dict[str, Union[str, List[str]]] = {
            'character': char,
            'codepoint': f"U+{ord(char):04X}",
            'definition_en': entry.get('definition') or entry.get('kDefinition', ''),
            'mandarin_pinyin': entry.get('cedict_pinyin') or entry.get('kMandarin', ''),
            'stroke_count': entry.get('kTotalStrokes', ''),
            'radical': entry.get('kRSUnicode', ''),
        }
        
        # Add source info
        sources: List[str] = []
        if char in unihan_data:
            sources.append('unihan')
        if char in cedict_data:
            sources.append('cedict')
        standardized['sources'] = sources
        
        merged[char] = standardized
    
    print(f"Merged metadata for {len(merged)} characters")
    return merged


def save_metadata_db(metadata: Dict[str, Dict[str, Union[str, List[str]]]], output_path: str = "data/char_metadata.json"):
    """
    Save character metadata to JSON file.
    
    Args:
        metadata: Character metadata dictionary
        output_path: Path to save JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Saved metadata for {len(metadata)} characters to {output_path}")


def load_metadata_db(metadata_path: str = "data/char_metadata.json") -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """
    Load character metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata JSON file
    
    Returns:
        Character metadata dictionary
    """
    metadata_file = Path(metadata_path)
    
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_path}")
        return {}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Loaded metadata for {len(metadata)} characters from {metadata_path}")
    return metadata


def get_character_metadata(char: str, metadata_db: Dict[str, Dict[str, Union[str, List[str]]]]) -> Optional[Dict[str, Union[str, List[str]]]]:
    """
    Get metadata for a single character.
    
    Args:
        char: Chinese character
        metadata_db: Loaded metadata database
    
    Returns:
        Character metadata or None if not found
    """
    return metadata_db.get(char)


def generate_caption(char: str, metadata: Optional[Dict[str, Union[str, List[str]]]] = None) -> str:
    """
    Generate a training caption for a character.
    
    Args:
        char: Chinese character
        metadata: Optional metadata dictionary
    
    Returns:
        Caption string for Stable Diffusion training
    """
    if metadata:
        codepoint = metadata.get('codepoint', f"U+{ord(char):04X}")
        definition = metadata.get('definition_en', '')
        pinyin = metadata.get('mandarin_pinyin', '')
        strokes = metadata.get('stroke_count', '')
        
        parts = [f"Chinese character {char} ({codepoint})"]
        
        if definition:
            parts.append(f"meaning '{definition}'")
        if pinyin:
            parts.append(f"pronounced '{pinyin}'")
        if strokes:
            parts.append(f"stroke count: {strokes}")
        
        return ", ".join(parts)
    else:
        return f"Chinese character {char} (U+{ord(char):04X})"


if __name__ == "__main__":
    # Example: Download and build metadata database
    print("Building character metadata database...")
    print("=" * 60)
    
    # Step 1: Download Unihan (if needed)
    try:
        download_unihan_data()
    except Exception as e:
        print(f"Could not download Unihan: {e}")
        print("Continuing with existing data if available...")
    
    # Step 2: Load Unihan data
    unihan_data = load_unihan_database()
    
    # Step 3: Download CC-CEDICT (if needed)
    try:
        download_cedict()
    except Exception as e:
        print(f"Could not download CC-CEDICT: {e}")
        print("Continuing with Unihan data only...")
    
    # Step 4: Parse CC-CEDICT
    cedict_path = Path("data/cedict.txt")
    cedict_data = {}
    if cedict_path.exists():
        cedict_data = parse_cedict(cedict_path)
    
    # Step 5: Merge and save
    if unihan_data or cedict_data:
        merged = merge_character_metadata(unihan_data, cedict_data)
        save_metadata_db(merged)
        
        # Test with a few characters
        test_chars = ["一", "人", "大", "中", "国"]
        print("\n" + "=" * 60)
        print("Sample metadata:")
        for char in test_chars:
            if char in merged:
                meta = merged[char]
                caption = generate_caption(char, meta)
                print(f"\n{char}: {caption}")
    else:
        print("No metadata sources available. Please download Unihan or CC-CEDICT first.")
