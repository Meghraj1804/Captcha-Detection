from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import string, random, os

# os.makedirs("synthetic_data", exist_ok=True)

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def add_noise(img):
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, img.width), random.randint(0, img.height)
        x2, y2 = random.randint(0, img.width), random.randint(0, img.height)
        draw.line([x1, y1, x2, y2], fill=random_color(), width=1)
    return img

def generate_captcha_images(count=10000):
    
    image_gen = ImageCaptcha(width=200, height=50)
    characters = string.ascii_letters + string.digits

    for i in range(count):
        captcha_text = ''.join(random.choices(characters, k=5))
        img = image_gen.generate_image(captcha_text).convert("RGB")
        
        # Rotate
        # img = img.rotate(random.uniform(-20, 20), resample=Image.BILINEAR, fillcolor=(255, 255, 255))
        
        # Add noise
        img = add_noise(img)
        
        # Optionally blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        img.save(f'raw_data/{captcha_text}.png')

generate_captcha_images(5000)
