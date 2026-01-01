"""Benchmark FLUX.1 PyGPUkit vs Diffusers.

Compares transformer inference time between:
1. PyGPUkit FluxTransformer (with native CUDA kernels)
2. Diffusers FluxTransformer2DModel (PyTorch)

Both use the same VAE and text encoders for fair comparison.
"""

import time
from pathlib import Path

import numpy as np

# Model paths
PYGPUKIT_MODEL_PATH = "F:/ImageGenerate/flux1-schnell-full"
DIFFUSERS_MODEL_PATH = (
    "F:/ImageGenerate/flux1-schnell-full/"
    "models--black-forest-labs--FLUX.1-schnell/snapshots/"
    "741f7c3ce8b383c54771c7003378a50191e9efe9"
)


def benchmark_pygpukit(
    model_path: str,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_steps: int = 4,
    warmup: int = 1,
    runs: int = 3,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Benchmark PyGPUkit FLUX implementation.

    Returns:
        Tuple of (average_time_ms, generated_image_array)
    """
    from pygpukit.core.factory import from_numpy
    from pygpukit.diffusion.models.flux.pipeline import FluxPipeline

    print("Loading PyGPUkit pipeline...")
    pipe = FluxPipeline.from_pretrained(model_path)

    # Pre-encode prompt (shared overhead)
    pooled_embed, t5_embed = pipe.encode_prompt(prompt)

    # Prepare inputs
    latent_h = height // 16
    latent_w = width // 16
    latent_seq_len = latent_h * latent_w

    from pygpukit.diffusion.models.flux.embeddings import prepare_image_ids, prepare_text_ids
    img_ids = prepare_image_ids(1, latent_h, latent_w)
    txt_ids = prepare_text_ids(1, t5_embed.shape[1])

    np.random.seed(seed)
    latents_np = np.random.randn(1, latent_seq_len, 64).astype(np.float32)

    def run_inference():
        """Run single inference pass with scheduler reset."""
        pipe.scheduler.set_timesteps(num_steps)  # Reset scheduler each time
        latents = latents_np.copy()
        for t in pipe.scheduler.timesteps:
            timestep = np.array([t], dtype=np.float32)
            noise_pred = pipe.transformer.forward(
                hidden_states=from_numpy(latents),
                encoder_hidden_states=from_numpy(t5_embed.astype(np.float32)),
                pooled_projections=from_numpy(pooled_embed.astype(np.float32)),
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
            ).to_numpy()
            latents = pipe.scheduler.step(noise_pred, t, latents)
        return latents

    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        latents = run_inference()

    # Benchmark
    print(f"Benchmarking ({runs} runs)...")
    times = []
    for i in range(runs):
        start = time.perf_counter()
        latents = run_inference()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f} ms")

    avg_time = sum(times) / len(times)

    # Decode final image
    image_np = pipe.decode_latents(latents, height, width)

    return avg_time, image_np[0]


def benchmark_diffusers(
    model_path: str,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_steps: int = 4,
    warmup: int = 1,
    runs: int = 3,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Benchmark Diffusers FluxPipeline.

    Returns:
        Tuple of (average_time_ms, generated_image_array)
    """
    import torch
    from diffusers import FluxPipeline

    print("Loading Diffusers pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    ).to(device)

    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        generator = torch.Generator(device=device).manual_seed(seed)
        _ = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

    # Benchmark
    print(f"Benchmarking ({runs} runs)...")
    times = []
    for i in range(runs):
        generator = torch.Generator(device=device).manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator,
        )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f} ms")

    avg_time = sum(times) / len(times)
    image = result.images[0]

    return avg_time, np.array(image)


def main():
    prompt = "A cute orange cat sitting on green grass, sunny day, photorealistic"
    height = 512
    width = 512
    num_steps = 4
    seed = 42

    print("=" * 60)
    print("FLUX.1 Schnell Benchmark: PyGPUkit vs Diffusers")
    print("=" * 60)
    print(f"PyGPUkit model: {PYGPUKIT_MODEL_PATH}")
    print(f"Diffusers model: {DIFFUSERS_MODEL_PATH}")
    print(f"Prompt: {prompt}")
    print(f"Size: {width}x{height}")
    print(f"Steps: {num_steps}")
    print("=" * 60)

    # Test PyGPUkit first
    print("\n[PyGPUkit]")
    try:
        pygpukit_time, pygpukit_img = benchmark_pygpukit(
            PYGPUKIT_MODEL_PATH, prompt, height, width, num_steps, seed=seed
        )
        print(f"Average time: {pygpukit_time:.1f} ms")

        from PIL import Image
        Image.fromarray(pygpukit_img).save("flux_pygpukit.png")
        print("Saved: flux_pygpukit.png")
    except Exception as e:
        print(f"PyGPUkit FAILED: {e}")
        import traceback
        traceback.print_exc()
        pygpukit_time = None
        pygpukit_img = None

    # Test Diffusers
    print("\n[Diffusers]")
    try:
        diffusers_time, diffusers_img = benchmark_diffusers(
            DIFFUSERS_MODEL_PATH, prompt, height, width, num_steps, seed=seed
        )
        print(f"Average time: {diffusers_time:.1f} ms")

        from PIL import Image
        Image.fromarray(diffusers_img).save("flux_diffusers.png")
        print("Saved: flux_diffusers.png")
    except Exception as e:
        print(f"Diffusers FAILED: {e}")
        import traceback
        traceback.print_exc()
        diffusers_time = None
        diffusers_img = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if pygpukit_time is not None:
        print(f"PyGPUkit:  {pygpukit_time:.1f} ms ({num_steps} steps)")
    else:
        print("PyGPUkit:  FAILED")

    if diffusers_time is not None:
        print(f"Diffusers: {diffusers_time:.1f} ms ({num_steps} steps)")
    else:
        print("Diffusers: FAILED")

    if pygpukit_time is not None and diffusers_time is not None:
        speedup = diffusers_time / pygpukit_time
        if speedup > 1:
            print(f"PyGPUkit is {speedup:.2f}x faster")
        else:
            print(f"Diffusers is {1/speedup:.2f}x faster")


if __name__ == "__main__":
    main()
