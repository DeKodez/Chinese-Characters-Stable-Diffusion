"""
Utility to convert Chinese Unicode characters to images.
This creates training data or reference images for Stable Diffusion.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import unicodedata
from pathlib import Path
from typing import Optional, List, Union


def get_platform_font_paths(platform: str = "macos") -> List[str]:
    """
    Get default Chinese font paths for a platform.
    
    Args:
        platform: "macos" or "windows"
    
    Returns:
        List of font paths to try
    """
    if platform.lower() == "windows":
        return [
            "C:/Windows/Fonts/simsun.ttc",  # SimSun
            "C:/Windows/Fonts/simhei.ttf",  # SimHei
            "C:/Windows/Fonts/msyh.ttc",    # Microsoft YaHei
            "C:/Windows/Fonts/msyhbd.ttc",  # Microsoft YaHei Bold
            "C:/Windows/Fonts/simkai.ttf",  # KaiTi
            "C:/Windows/Fonts/STKAITI.TTF", # STKaiti
        ]
    else:  # macOS (default)
        return [
            "/System/Library/Fonts/Supplemental/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/STHeiti Medium.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Microsoft/SimHei.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]


def get_chinese_font(font_size=256, platform: str = "macos", font_paths: Optional[List[str]] = None):
    """
    Try to find a Chinese font on the system.
    Falls back to default font if none found.
    
    Args:
        font_size: Size of the font
        platform: "macos" or "windows" (used if font_paths is None)
        font_paths: Custom list of font paths to try (overrides platform)
    
    Returns:
        ImageFont object
    """
    # Use custom font paths if provided, otherwise use platform defaults
    if font_paths is None:
        font_paths = get_platform_font_paths(platform)
    
    # Try to load a Chinese font
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except (OSError, IOError):
                continue
    
    # Fallback to default font (may not render Chinese properly)
    if platform.lower() == "windows":
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except (OSError, IOError):
            return ImageFont.load_default()
    else:  # macOS
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            return ImageFont.load_default()


def char_to_image(
    char: str,
    output_path: Optional[str] = None,
    size: tuple = (512, 512),
    font_size: int = 400,
    bg_color: str = "white",
    text_color: str = "black",
    platform: str = "macos",
    font_paths: Optional[List[str]] = None
):
    """
    Convert a Chinese character to an image.
    
    Args:
        char: The Chinese character (Unicode string)
        output_path: Where to save the image (optional)
        size: Image dimensions (width, height)
        font_size: Size of the font
        bg_color: Background color
        text_color: Text color
        platform: "macos" or "windows" (default: "macos")
        font_paths: Custom list of font paths to try (overrides platform)
    
    Returns:
        PIL Image object
    """
    # Create image with background
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Get font
    font = get_chinese_font(font_size, platform=platform, font_paths=font_paths)
    
    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2 - bbox[1]
    
    # Draw the character
    draw.text((x, y), char, font=font, fill=text_color)
    
    # Save if path provided
    if output_path:
        img.save(output_path)
        print(f"Saved: {output_path}")
    
    return img


def generate_char_dataset(
    chars: list,
    output_dir: str = "char_images",
    size: tuple = (512, 512),
    font_size: int = 400,
    platform: str = "macos",
    font_paths: Optional[List[str]] = None
):
    """
    Generate images for a list of Chinese characters.
    
    Args:
        chars: List of Chinese characters
        output_dir: Directory to save images
        size: Image dimensions
        font_size: Font size
        platform: "macos" or "windows" (default: "macos")
        font_paths: Custom list of font paths to try (overrides platform)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for char in chars:
        # Use Unicode code point as filename
        unicode_code = f"U+{ord(char):04X}"
        filename = f"{unicode_code}_{char}.png"
        filepath = output_path / filename
        
        char_to_image(
            char,
            str(filepath),
            size=size,
            font_size=font_size,
            platform=platform,
            font_paths=font_paths
        )
    
    print(f"\nGenerated {len(chars)} character images in {output_dir}/")


def parse_unicode_range(start: Union[str, int], end: Union[str, int]) -> tuple:
    """
    Parse Unicode range strings (e.g., "U+4E00") to integers.
    
    Args:
        start: Start Unicode code point (e.g., "U+4E00" or "0x4E00" or 19968)
        end: End Unicode code point (e.g., "U+9FFF" or "0x9FFF" or 40959)
    
    Returns:
        Tuple of (start_code, end_code) as integers
    """
    def parse_code_point(code: Union[str, int]) -> int:
        if isinstance(code, int):
            return code
        code = code.strip().upper()
        if code.startswith("U+"):
            return int(code[2:], 16)
        elif code.startswith("0X"):
            return int(code[2:], 16)
        elif code.startswith("0x"):
            return int(code[2:], 16)
        else:
            # Try as hex first, then decimal
            try:
                return int(code, 16)
            except ValueError:
                return int(code)
    
    start_code = parse_code_point(start)
    end_code = parse_code_point(end)
    
    if start_code > end_code:
        raise ValueError(f"Start code point ({start_code}) must be <= end code point ({end_code})")
    
    return start_code, end_code


def generate_unicode_range_dataset(
    start: Union[str, int],
    end: Union[str, int],
    output_dir: str = "char_images",
    size: tuple = (512, 512),
    font_size: int = 400,
    platform: str = "macos",
    font_paths: Optional[List[str]] = None,
    skip_invalid: bool = True,
    progress_interval: int = 100
):
    """
    Generate images for all characters in a Unicode range.
    
    Args:
        start: Start Unicode code point (e.g., "U+4E00", "0x4E00", or 19968)
        end: End Unicode code point (e.g., "U+9FFF", "0x9FFF", or 40959)
        output_dir: Directory to save images
        size: Image dimensions
        font_size: Font size
        platform: "macos" or "windows" (default: "macos")
        font_paths: Custom list of font paths to try (overrides platform)
        skip_invalid: Skip invalid/unassigned code points (default: True)
        progress_interval: Print progress every N characters (default: 100)
    
    Returns:
        Number of characters successfully generated
    """
    # Parse Unicode range
    start_code, end_code = parse_unicode_range(start, end)
    
    print(f"Generating images for Unicode range U+{start_code:04X} to U+{end_code:04X}")
    print(f"Total range: {end_code - start_code + 1} code points")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    generated_count = 0
    skipped_count = 0
    
    for code_point in range(start_code, end_code + 1):
        try:
            # Convert code point to character
            char = chr(code_point)
            
            # Check if character is valid and printable
            if skip_invalid:
                # Skip control characters, surrogates, and private use areas
                if not char.isprintable() or char.isspace():
                    skipped_count += 1
                    continue
                
                # Check if character has a valid Unicode category
                try:
                    category = unicodedata.category(char)
                    # Skip control, format, private use, and unassigned characters
                    if category.startswith(('C', 'Z', 'Co', 'Cn')):
                        skipped_count += 1
                        continue
                except (ValueError, TypeError):
                    skipped_count += 1
                    continue
            
            # Generate filename
            unicode_code = f"U+{code_point:04X}"
            filename = f"{unicode_code}_{char}.png"
            filepath = output_path / filename
            
            # Generate image
            char_to_image(
                char,
                str(filepath),
                size=size,
                font_size=font_size,
                platform=platform,
                font_paths=font_paths
            )
            
            generated_count += 1
            
            # Progress update
            if generated_count % progress_interval == 0:
                print(f"Progress: {generated_count} characters generated, {skipped_count} skipped")
        
        except (ValueError, OverflowError) as e:
            # Invalid code point
            if not skip_invalid:
                print(f"Warning: Skipping invalid code point U+{code_point:04X}: {e}")
            skipped_count += 1
            continue
        except (OSError, IOError) as e:
            print(f"Error processing U+{code_point:04X}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nCompleted! Generated {generated_count} character images in {output_dir}/")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid/unprintable characters")
    
    return generated_count


if __name__ == "__main__":
    # Example: Generate images for some common Chinese characters
    example_chars = ["人", "大", "小", "中", "国", "爱", "学", "生"]
    
    print("Generating example character images...")
    generate_char_dataset(example_chars, "char_images")
    
    # Single character example
    print("\nGenerating single character example...")
    char_to_image("字", "example_char.png", size=(512, 512))

    # generate_unicode_range_dataset("U+4E00", "U+9FFF", "cjk_ideographs", progress_interval=500)
