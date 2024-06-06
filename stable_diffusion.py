from diffusers import DiffusionPipeline
import torch
# prompt = "A majestic lion walking down a forest while a small, braying ass watches from the side, both surrounded by a respectful audience of various forest animals."
# prompt = "realistic tiger running in the forest"
# prompt = "Sarah driving her old Mustang at night on a deserted road, with the windows down and the soft glow of the dashboard illuminating her face as the radio plays quietly."
# prompt = "Generate an image of Sarah driving her old Mustang on a deserted road at night, with the windows down and the radio playing softly"
# prompt = "Generate an image of Jack walking with his Golden Retriever through a park at sunset, reflecting a serene evening ambiance."
# prompt = "Generate an image of Mark and his friends warming up with stew from a Dutch oven by a campfire in the mountains."
# prompt = "Generate an image of Sarah setting sail in her small sailboat on the calm waters of the bay, feeling exhilarated by the early morning breeze."
# prompt = "Generate an image of a donkey making a loud noise inside a cave, causing goats to run out, while a lion waits at the entrance, ready to pounce, set in a wild, natural environment."
prompt = "Generate an image of a fox and a lion."
# pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
# ).to("cuda")
pipeline_text2image = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=False, variant="fp16")
pipeline_text2image.to("cuda")
# prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt,guidance_scale = 20,num_inference_steps=200).images[0]
image.save(f"foxlion.png")
