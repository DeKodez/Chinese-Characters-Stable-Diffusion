"""
Utility to convert Chinese Unicode characters to images.
This creates training data or reference images for Stable Diffusion.
"""

from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
from typing import Optional, List


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


if __name__ == "__main__":
    # Example: Generate images for some common Chinese characters
    example_chars = ["人", "大", "小", "中", "国", "爱", "学", "生"]
    
    print("Generating example character images...")
    generate_char_dataset(example_chars, "char_images")
    
    # Single character example
    print("\nGenerating single character example...")
    char_to_image("字", "example_char.png", size=(512, 512))