from huggingface_hub import login
login(token="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")

# Step 3: Load the Stable Diffusion model
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Step 4: Ask user for input prompt (in Colab-friendly way)
from IPython.display import display
import ipywidgets as widgets

prompt_input = widgets.Text(
    value='A man eating a fish',
    placeholder='Type your prompt here',
    description='Prompt:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='100%')
)

display(prompt_input)

# Run this cell AFTER entering your prompt above
# Step 5: Generate image based on user input
def generate_image(prompt):
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    display(image)

generate_image(prompt_input.value)

# Step 6: Download image
from google.colab import files
files.download("generated_image.png")