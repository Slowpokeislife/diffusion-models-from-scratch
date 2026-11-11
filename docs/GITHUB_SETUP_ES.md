# Publicar en GitHub y preparar el portafolio

## 1. Qué subir

Sube el contenido de esta carpeta `diffusion-models-rebuilt/` (README, `src`, `scripts`, `tests`, `docs`, `demo`, `requirements.txt`, `.gitignore`, `LICENSE`). **No subas** `Flores.gz`, la carpeta `data`, checkpoints privados, ni notebooks oficiales DLI sin revisar permisos del curso.

## 2. Comandos de Git en PowerShell

```powershell
cd "C:\ruta\a\diffusion-models-rebuilt"
git init
git add .
git commit -m "Reconstruct diffusion model experiments with PyTorch"
git branch -M main
git remote add origin https://github.com/Slowpokeislife/diffusion-models-from-scratch.git
git push -u origin main
```

Antes del `git push`, crea un repositorio vacío con el nombre `diffusion-models-from-scratch` en GitHub. Si Git ya estaba inicializado, omite `git init`. Revisa `git status` para verificar que no aparezcan datos ni pesos.

**Repository About:** `Educational DDPM image generation with PyTorch, conditional U-Net, classifier-free guidance, and diffusion GIF/MP4 visualizations. Independently reconstructed after NVIDIA DLI coursework.`

**Topics:** `pytorch`, `diffusion-models`, `ddpm`, `generative-ai`, `computer-vision`, `classifier-free-guidance`, `unet`, `deep-learning`, `fashion-mnist`, `mnist`, `image-generation`.

## 3. Resultados nuevos que conviene generar

Ejecuta `python scripts/forward.py` para visualizar la corrupción progresiva, después entrena y ejecuta `sample.py`. Compara los GIF originales con los regenerados sin decir que son idénticos. Si un entrenamiento produce imágenes indistinguibles de ruido, publícalo como resultado inicial/limitación, no como convergencia exitosa.

Para la página de tu portafolio, mantén una tarjeta `Generative AI · Diffusion Models`, con una galería MP4/GIF y estas secciones: Contexto (formación NVIDIA), Problema (síntesis de imágenes), Arquitectura U-Net, Proceso forward/reverse, Condicionamiento CFG, Demostraciones originales y nuevas, Aprendizajes y Limitaciones.

## 4. Sobre los notebooks originales

En el paquete privado separado se incluyen notebooks del curso con ejercicios `FIXME` corregidos. No son una entrega histórica verificada ni deben presentarse como código original rescatado. No se recomienda subirlos públicamente sin verificar condiciones del curso y licencias.
