# End-to-end workflow | Reconstructed diffusion experiments

## 0. What was recovered vs. what was reconstructed?

**Recovered:** `fashion.mp4`, `flowers.mp4`, `fashionMNIST.gif`, `flowers.gif`, and the original compressed course exercise archive. **Not recovered:** original trained model parameters, training logs, the exact code submissions, experiment configuration or saved seeds. The preserved clips are evidence of course-time experiments, **not evidence that this repository reproduces the exact frames**.

**Reconstructed here:** independent source code for dataset loading, U-Net, forward and reverse diffusion, classifier-free guidance, training, sampling, export, tests and optional CLIP image/text similarity. The notebook `06_Assessment.ipynb` is resolved in the private reference archive, but it does not establish that those exact edits were present in your historic submission.

## 1. Data

- Fashion-MNIST: grayscale clothing images; `torchvision.datasets.FashionMNIST` downloads it.
- MNIST: grayscale handwritten digits; downloads through torchvision.
- Flowers: RGB flower images. The old cropped data archive contains daisy, rose and sunflower folders. To reuse locally, extract it under `data/cropped_flowers/`. Data are not bundled in the public repo.
- Images are resized to 32×32; all pixels are normalized to [-1, 1]. The original course notebooks used 16×16 Fashion-MNIST and 28×28 MNIST in some stages; this independent implementation standardizes all datasets at 32×32 to support repeated downsampling.

## 2. Denoising autoencoder and U-Net fundamentals

A U-Net processes an input through convolutional blocks, downsamples to extract larger receptive-field features, then upsamples back to pixel space. Skip connections preserve localization and high-frequency cues. The model predicts **noise** epsilon, not the clean image directly. This reconstruction uses residual blocks, GroupNorm, SiLU/GELU-style smooth nonlinearities and sinusoidal time features, with class embeddings.

## 3. Forward diffusion

Let beta_t be the noise variance schedule; alpha_t = 1 - beta_t, and alpha_bar_t = product(alpha_1...alpha_t). A direct sample at timestep t is:

`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon`, epsilon ~ N(0, I).

**Produces a video:** `python scripts/forward.py --dataset fashion --output demo/forward_fashion` generates `forward_fashion.gif` and `forward_fashion.mp4`. `--dataset flowers` produces its RGB counterpart, after installing the flowers dataset.

## 4. Training objective

Choose a batch of real images and independently sampled t. Add noise with the closed-form equation. Train the U-Net to estimate that sampled noise using mean-squared error: `L = E[||epsilon - epsilon_theta(x_t,t,c)||²]`. Record average training loss by epoch. A falling loss is useful but not itself proof of realistic samples.

The dataloader handles partial final batches; a class label is replaced with the null category independently per image with probability `drop_prob` (default 0.1).

## 5. Reverse diffusion

Start from random Gaussian noise. At each timestep predict epsilon and use the DDPM reverse mean:

`mu_t = (x_t - beta_t * epsilon_theta / sqrt(1-alpha_bar_t)) / sqrt(alpha_t)`.

Add posterior Gaussian noise at timesteps above zero, with variance `beta_t * (1-alpha_bar_(t-1)) / (1-alpha_bar_t)`. At t=0 don't add noise. The code stores a configurable number of intermediate frames.

## 6. Classifier-free guidance (CFG)

During training, sometimes omit the class label; at generation, compare conditional and unconditional noise estimates:

`epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)`.

Here `w=1` means ordinary conditional prediction. Larger weights can sharpen class adherence but can distort/diversify samples. `--guidance 2.0` is a starting point, **not a validated optimum**.

## 7. Rebuild videos

1. Train `python scripts/train.py --dataset fashion --epochs 20`.
2. Generate `python scripts/sample.py --checkpoint checkpoints/fashion.pt --samples 70 --output demo/generated_fashion`.
3. Prepare flower images and train `python scripts/train.py --dataset flowers --epochs 100 --batch-size 32`.
4. Generate `python scripts/sample.py --checkpoint checkpoints/flowers.pt --samples 21 --guidance 2 --output demo/generated_flowers`.
5. Optionally train MNIST, generate digits and compare the three tasks. The samples and animation will be **new**, not frame-identical to saved course videos.

These calls produce **PNG**, **GIF**, **MP4** for each reverse denoising run. Forward animation produces GIF+MP4. Run with a capable GPU for practical training; CPU tests are sufficient to validate functionality but training will be slow.

## 8. Optional CLIP experiment (separate from image generation)

Install `pip install transformers`, then:

`python scripts/clip_similarity.py --image data/cropped_flowers/daisy/your_photo.jpg --texts "a daisy" "a sunflower" "a rose"`

CLIP compares text/image representations; this **does not mean** the reconstruction implements text-to-image guidance, which would require additional architecture and training modifications.

## 9. Evaluation and limitations

- Verify MSE curves and inspect class-conditioned generated grids over training.
- Compare unconditional (`w=0`) and conditional (`w=1,2,4`) runs with the same seed.
- Document hardware, dataset version, batch size, epochs, and actual outcome before making performance claims.
- These images are low resolution (32×32), and visually recognizable generations are not guaranteed with the default epoch budgets. Training results depend strongly on compute and tuning.
- No pretrained diffusion checkpoints or actual training metrics were recovered.
