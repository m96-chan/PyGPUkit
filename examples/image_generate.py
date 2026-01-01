#!/usr/bin/env python3
"""Image Generation Example using PyGPUkit Diffusion.

This example demonstrates text-to-image generation using:
- Stable Diffusion 3 (SD3)
- Flux.1 (Schnell/Dev)
- PixArt-Sigma

Usage:
    # Demo mode (no model required, generates random patterns)
    python examples/image_generate.py --demo

    # With actual model
    python examples/image_generate.py --model F:/SD3/sd3-medium --prompt "A cat"

    # Flux model
    python examples/image_generate.py --model F:/Flux/flux1-schnell --type flux

Requirements:
    - PyGPUkit (pip install -e .)
    - Pillow for image saving
    - scipy for VAE interpolation (optional)
    - tokenizers for text encoding (optional)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def demo_mode(args: argparse.Namespace) -> None:
    """Run demo mode with random weights."""
    from pygpukit.diffusion import Text2ImagePipeline

    print("=" * 60)
    print("PyGPUkit Image Generation Demo")
    print("=" * 60)
    print()
    print("Running in DEMO mode (no model weights required)")
    print("This will generate random noise patterns to test the pipeline.")
    print()

    # Create demo pipeline
    model_type = args.type or "sd3"
    print(f"Creating {model_type.upper()} demo pipeline...")
    pipe = Text2ImagePipeline.create_demo_pipeline(model_type=model_type)

    # Generate image
    prompt = args.prompt or "A beautiful sunset over mountains"
    print(f"Prompt: {prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}")
    print()

    start_time = time.time()

    def progress_callback(step: int, total: int, latents):
        elapsed = time.time() - start_time
        print(f"  Step {step + 1}/{total} ({elapsed:.1f}s)")

    print("Generating image...")
    image = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        callback=progress_callback,
    )

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.2f}s")

    # Save image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved to: {output_path}")

    print()
    print("NOTE: Demo mode generates random patterns, not actual images.")
    print("      For real image generation, provide a model path with --model")


def load_and_generate(args: argparse.Namespace) -> None:
    """Load model and generate image."""
    from pygpukit.diffusion import Text2ImagePipeline

    print("=" * 60)
    print("PyGPUkit Image Generation")
    print("=" * 60)
    print()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        print()
        print("Please provide a valid model path. Supported models:")
        print("  - Stable Diffusion 3 (sd3-medium, sd3-large)")
        print("  - Flux.1 (flux1-schnell, flux1-dev)")
        print("  - PixArt-Sigma")
        print()
        print("Example model paths:")
        print("  F:/SD3/sd3-medium/")
        print("  F:/Flux/flux1-schnell.safetensors")
        return

    print(f"Loading model from: {model_path}")
    print(f"Model type: {args.type or 'auto-detect'}")
    print()

    start_load = time.time()
    pipe = Text2ImagePipeline.from_pretrained(
        model_path,
        dtype=args.dtype,
        model_type=args.type,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Generate image
    prompt = args.prompt or "A beautiful landscape with mountains and a river"
    print(f"Prompt: {prompt}")
    if args.negative_prompt:
        print(f"Negative: {args.negative_prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}")
    print(f"CFG Scale: {args.guidance_scale}")
    print(f"Seed: {args.seed or 'random'}")
    print()

    start_gen = time.time()

    def progress_callback(step: int, total: int, latents):
        elapsed = time.time() - start_gen
        remaining = (elapsed / (step + 1)) * (total - step - 1)
        print(f"  Step {step + 1}/{total} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")

    print("Generating image...")
    image = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        callback=progress_callback,
    )

    gen_time = time.time() - start_gen
    print(f"\nGeneration complete in {gen_time:.2f}s")
    print(f"  ({gen_time / args.steps:.3f}s per step)")

    # Save image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"\nSaved to: {output_path}")


def batch_generate(args: argparse.Namespace) -> None:
    """Generate multiple images with different prompts."""
    from pygpukit.diffusion import Text2ImagePipeline

    prompts = [
        "A serene Japanese garden with cherry blossoms",
        "A cyberpunk city at night with neon lights",
        "A cozy cabin in snowy mountains",
        "An underwater coral reef with colorful fish",
    ]

    print("=" * 60)
    print("PyGPUkit Batch Image Generation")
    print("=" * 60)
    print()

    # Create demo pipeline if no model specified
    if args.model:
        pipe = Text2ImagePipeline.from_pretrained(args.model, dtype=args.dtype)
    else:
        print("Using demo pipeline (random patterns)")
        pipe = Text2ImagePipeline.create_demo_pipeline()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] {prompt[:50]}...")

        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed + i if args.seed else None,
        )

        output_path = output_dir / f"image_{i + 1:02d}.png"
        image.save(output_path)
        print(f"  Saved: {output_path}")

    print(f"\nBatch generation complete! {len(prompts)} images saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate images using PyGPUkit Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no model required)
  python examples/image_generate.py --demo

  # Generate with SD3
  python examples/image_generate.py --model F:/SD3/sd3-medium --prompt "A cat"

  # Generate with Flux
  python examples/image_generate.py --model F:/Flux/flux1-schnell --type flux

  # Batch generation
  python examples/image_generate.py --batch --demo
""",
    )

    # Mode selection
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode without model (generates random patterns)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate batch of images with different prompts",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model directory or safetensors file",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["sd3", "flux", "pixart"],
        default=None,
        help="Model type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Weight dtype (default: float32)",
    )

    # Generation settings
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted",
        help="Negative prompt for CFG",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height (default: 1024)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width (default: 1024)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps (default: 28)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="CFG guidance scale (default: 7.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default="output/generated.png",
        help="Output image path (default: output/generated.png)",
    )

    args = parser.parse_args()

    # Run appropriate mode
    try:
        if args.batch:
            batch_generate(args)
        elif args.demo or args.model is None:
            demo_mode(args)
        else:
            load_and_generate(args)
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
