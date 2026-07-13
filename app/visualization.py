import io
import base64
from PIL import Image, ImageDraw, ImageFont

def draw_detections_on_image(image: Image.Image, detection_data: list) -> Image.Image:
    """
    Draws bounding boxes and labels on the image for each detected bacterium.
    
    Green color code for G+ (Gram Positive)
    Red color code for G- (Gram Negative)
    """
    # Create a copy to draw on, preventing modifications to the original cached/referenced image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to load a font, fallback to default
    font = None
    font_sizes = [16, 14, 12]
    for size in font_sizes:
        try:
            # Try to load standard fonts on Windows/Linux
            font = ImageFont.truetype("arial.ttf", size)
            break
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size)
                break
            except (OSError, IOError):
                continue
                
    if font is None:
        font = ImageFont.load_default()

    for detection in detection_data:
        box = detection["box"]  # [x1, y1, x2, y2]
        class_label = detection["class"]  # "G+" or "G-"
        confidence = detection["confidence"]
        
        # Color coding: Green for G+, Red for G-
        if class_label == "G+":
            # Premium Green
            color = (46, 204, 113)  # Emerald green
        else:
            # Premium Red
            color = (231, 76, 60)   # Alizarin red
            
        # Draw bounding box
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        # format confidence as percentage
        conf_val = confidence if confidence <= 1.0 else confidence / 100.0
        label = f"{class_label} ({conf_val:.1%})"
        
        # Calculate label background position
        try:
            # Pillow >= 10.0
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            # Fallback for older Pillow
            if hasattr(draw, 'textsize'):
                text_width, text_height = draw.textsize(label, font=font)
            else:
                text_width = len(label) * 8
                text_height = 12

        # Draw label background block just above the box (or inside if too close to top)
        label_y1 = y1 - text_height - 6
        label_y2 = y1
        if label_y1 < 0:
            label_y1 = y2
            label_y2 = y2 + text_height + 6
            
        draw.rectangle([x1, label_y1, x1 + text_width + 8, label_y2], fill=color)
        
        # Draw label text
        draw.text((x1 + 4, label_y1 + 2), label, fill=(255, 255, 255), font=font)
        
    return annotated_image

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    # Save as JPEG by default; if format is RGBA, save as PNG to avoid transparency issues
    save_format = format
    if image.mode in ("RGBA", "P") and format == "JPEG":
        save_format = "PNG"
    image.save(buffered, format=save_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Converts a PIL Image to raw bytes."""
    buffered = io.BytesIO()
    save_format = format.upper()
    if image.mode in ("RGBA", "P") and save_format == "JPEG":
        save_format = "PNG"
    image.save(buffered, format=save_format)
    return buffered.getvalue()
