"""Optional zero-shot CLIP exploration. Install transformers and a compatible model.

This script compares image/text embeddings; it does NOT train the DDPM or
condition the project UNet on free-form text.
"""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--texts', nargs='+', default=['a daisy', 'a sunflower', 'a rose'])
    parser.add_argument('--model', default='openai/clip-vit-base-patch32')
    args = parser.parse_args()
    try:
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
    except ImportError as exc:
        raise SystemExit('Install optional dependencies: pip install transformers') from exc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    image = Image.open(Path(args.image)).convert('RGB')
    batch = processor(text=args.texts, images=image, return_tensors='pt', padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
        probabilities = output.logits_per_image.softmax(dim=-1)[0].cpu().tolist()
    for label, probability in sorted(zip(args.texts, probabilities), key=lambda pair: -pair[1]):
        print(f'{probability:.3f}  {label}')


if __name__ == '__main__':
    main()
