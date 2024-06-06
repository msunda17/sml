from diffusers import DiffusionPipeline
import torch
# prompt = "A majestic lion walking down a forest while a small, braying ass watches from the side, both surrounded by a respectful audience of various  forest animals."
# prompt = "realistic tiger running in the forest"
# prompt = "Sarah driving her old Mustang at night on a deserted road, with the windows down and the soft glow of the dashboard illuminating her face as the radio plays quietly."
# prompt = "Generate an image of Sarah driving her old Mustang on a deserted road at night, with the windows down and the radio playing softly"
# prompt = "Generate an image of Jack walking with his Golden Retriever through a park at sunset, reflecting a serene evening ambiance."
# prompt = "Generate an image of Mark and his friends warming up with stew from a Dutch oven by a campfire in the mountains."
# prompt = "Generate an image of Sarah setting sail in her small sailboat on the calm waters of the bay, feeling exhilarated by the early morning breeze."
# prompt = "Generate an image of a woman named Lily walking through the city streets with her rescue mutt, each step filled with determination and gratitude for their second chance at life together."
# prompt = "Generate an image of a woman driving a car on rugged mountain roads, with a map spread out on the passenger seat, navigating hairpin turns"
# prompt = "Generate an image of a man stuck in traffic on a busy city street, surrounded by honking horns and flashing lights, feeling frustrated."
# prompt ="Generate an image of a man with the top down in a convertible, cruising along a scenic countryside road, with rolling hills and lush greenery around him"
# prompt = "Generate an image of a man driving a car filled with camping gear, exploring an open road that stretches out before him, discovering new towns and roadside attractions"
# prompt = "Generate an image of a man driving a car with a trunk full of camping gear on an open road, exploring new towns and roadside attractions"
# prompt = "Generate an image of a man driving a car with a trunk full of camping gear on an open road, exploring new towns and roadside attractions"
# prompt = "Generate an image of a man driving through the familiar streets of his hometown, passing landmarks, seen through tear-filled eyes, filled with longing for the past."
prompt = "Generate an image of a man named James and his crew racing their sailboat in a regatta on choppy seas, battling fierce winds as they maneuver towards the finish line, driven by the thrill of competition."
# pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
# ).to("cuda")
pipeline_text2image = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=False, variant="fp16")
pipeline_text2image.to("cuda")
# prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt,guidance_scale = 12.5,num_inference_steps=200).images[0]
image.save(f"./sailing/sailing9.png")