from PIL import Image

bmp_image_path = '/scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/adversarial_images/prompt_constrained_16.bmp'

with Image.open(bmp_image_path) as img:
    jpg_image_path = bmp_image_path.replace('.bmp', '.jpg')
    img.save(jpg_image_path, 'JPEG')

