import torch
import cv2
import os
from pathlib import Path
from diffusers import DiffusionPipeline

pictures_dir = Path(pathlib.Path.home(), "Pictures/")
os.mkdir(str(pictures_dir)+"/generated_images")

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

n_steps = 40
high_noise_frac = 0.8

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = int(input("Enter the n_steps, now  – %i: " % (n_steps)) or n_steps)
high_noise_frac = float(input("Enter the high noice frac, now  – %f: " % (high_noise_frac)) or high_noise_frac)

print ("Let's write ur promt...") 
subject = input("\033[31m\033[47m1st step\033[0m – The subject is what you want to see in the image. \
\033[32m\033[47m#e.g.: man with big axe, jumping, screaming, detailed face.\033[0m:")

medium = input("\033[31m\033[47m2nd step\033[0m – The Medium is the material used to make artwork. \
\033[32m\033[47m#e.g.: illustration, oil painting, 3D rendering, and photography.\033[0m:")

style = input("\033[31m\033[47m3rd step\033[0m – The Style refers to the artistic style of the image. \
\033[32m\033[47m#e.g.: impressionist, surrealist, pop art.\033[0m:")

resolution = input("\033[31m\033[47m3rd step\033[0m – The Resolution represents how sharp and detailed the image is. \
\033[32m\033[47m#e.g.: highly detailed, sharp focus.\033[0m:")

details = input("\033[31m\033[47m3rd step\033[0m – The Additional details are sweeteners added to modify an image. \
\033[32m\033[47m#e.g.: [salvage: 0.4], sci-fi, [glasses: 0.8]\033[0m:")

color = input("\033[31m\033[47m3rd step\033[0m – The Color control the overall color of the image by adding color keywords. \
\033[32m\033[47m#e.g.: (grey: 0.3), (red: 0.5), (black:0.2)\033[0m:")

lighting = input("\033[31m\033[47m3rd step\033[0m – The Lighting keywords can have a huge effect on how the image looks. \
\033[32m\033[47m#e.g.: cinematic lighting, dark\033[0m:")
promt = subject + "," + medium + "," + style + "," + resolution + "," + details + "," + color + "," + lighting 

num_repeat_steps = input("enter the required number of images: ")

for index in range(0, num_inference_steps):     
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
    cv2.imwrite(str(pictures_dir)+"/generated_images/"+num_repeat_steps.png, image)    
