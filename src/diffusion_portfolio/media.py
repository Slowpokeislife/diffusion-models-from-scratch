"""Export multi-sample diffusion animations as GIF + MP4."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from torchvision.utils import make_grid
import imageio.v2 as imageio


def grid_image(batch, columns=10, tile=8):
    pixels = (batch.detach().float().cpu().clamp(-1, 1) + 1) / 2
    grid = make_grid(pixels, nrow=columns, padding=2, pad_value=0.5)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')
    if arr.shape[2] == 1:
        arr = arr.repeat(3, 2)
    img = Image.fromarray(arr, 'RGB')
    if tile != 1:
        img = img.resize((img.width * tile, img.height * tile), Image.Resampling.NEAREST)
    return img


def export_frames(frames, output_stem, columns=10, fps=8, tile=4):
    path = Path(output_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [grid_image(batch, columns, tile) for batch in frames]
    images[0].save(path.with_suffix('.gif'), save_all=True, append_images=images[1:],
                   duration=round(1000 / fps), loop=0, optimize=False)
    # MP4 H.264 requires even dimensions; codec typically best with multiples of 16.
    width = ((images[0].width + 15) // 16) * 16
    height = ((images[0].height + 15) // 16) * 16
    try:
        with imageio.get_writer(str(path.with_suffix('.mp4')), fps=fps, codec='libx264',
                                macro_block_size=16, quality=7) as writer:
            for image in images:
                canvas = Image.new('RGB', (width, height), 'white')
                canvas.paste(image)
                writer.append_data(np.asarray(canvas))
    except Exception as exc:
        raise RuntimeError('Could not export MP4; install imageio-ffmpeg') from exc
    return path.with_suffix('.gif'), path.with_suffix('.mp4')
