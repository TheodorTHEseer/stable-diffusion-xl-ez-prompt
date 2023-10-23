from diffusers import DiffusionPipeline
import torch

#stdout for users
print ("\033[31m\033[47mCheck ur py imports: diffusers, " )
print ("#pip install diffusers --upgrade")
print ("#pip install invisible_watermark transformers accelerate safetensors\033[0m")

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = int(input("Enter the n_steps, now  – 40: ") or "40")
high_noise_frac = float(input("Enter the high noice frac, now  – 0.8: ") or "0.8")

print ("Let's write ur promt...") 
subject = input("\033[31m\033[47m1st step\033[0m – The subject is what you want to see in the image. \
\033[32m\033[47m#e.g.: man with big axe, jumping, screaming, detailed face.\033[0m:")

medium = input("\033[31m\033[47m2nd step\033[0m – Medium is the material used to make artwork. \
\033[32m\033[47m#e.g.: illustration, oil painting, 3D rendering, and photography.\033[0m:")


promt = subject + medium + 

num_repeat_steps = input("enter the required number of images: ")

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

show(image)