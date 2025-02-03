import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from controlnet_aux import LineartDetector
from compel import Compel
import argparse

MODEL_NAME = "SG161222/Realistic_Vision_V5.0_noVAE"
MODEL_CACHE = "cache"
VAE_CACHE = "vae-cache"


class Predictor:
    def __init__(self):
        """Initialize and load models."""
        self.setup()

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient."""

        # Load multiple ControlNets
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        lineart_controlnet = ControlNetModel.from_pretrained(
            "ControlNet-1-1-preview/control_v11p_sd15_lineart", torch_dtype=torch.float16
        )
        self.lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

        controlnets = [controlnet, lineart_controlnet]

        # Create the pipeline
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            MODEL_NAME,
            controlnet=controlnets,
            torch_dtype=torch.float16,
        )

        # For prompt processing
        self.compel_proc = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder
        )

        self.pipe = pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

    def resize_image(self, image, max_width, max_height):
        """
        Resize an image to a specific height/width while maintaining aspect ratio.
        """
        original_width, original_height = image.size
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        resize_ratio = min(width_ratio, height_ratio)

        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def make_inpaint_condition(self, image, image_mask):
        """
        Convert image + mask into inpainting condition (pixels replaced with -1 where masked).
        """
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[:2] == image_mask.shape[:2], "Image and mask must have the same size."
        image[image_mask > 0.5] = -1.0  # Mark masked pixels
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def closest_multiple_of_8(self, width, height):
        """
        Rounds width/height up to the closest multiple of 8.
        """
        w = ((width + 7) // 8) * 8
        h = ((height + 7) // 8) * 8
        return w, h

    def predict(
        self,
        image: str,
        prompt: str = "(a tabby cat)+++, high resolution, sitting on a park bench",
        mask: str = None,
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        strength: float = 0.8,
        max_height: int = 612,
        max_width: int = 612,
        steps: int = 20,
        seed: int = None,
        guidance_scale: float = 10.0,
        out_path: str = None
    ) -> str:
        """
        image (str): Path to input image.
        mask (str): Path to mask image.
        prompt (str): Positive text prompt.
        negative_prompt (str): Negative text prompt.
        strength (float): Control strength/weight.
        max_height (int): Maximum allowable height.
        max_width (int): Maximum allowable width.
        steps (int): Number of denoising steps.
        seed (int): Random seed (if None, random).
        guidance_scale (float): Guidance scale.

        Returns:
            str: File path to the output image (/tmp/output.png).
        """

        # Handle random seed
        if not seed or seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)

        # Load and resize images
        init_image = Image.open(image)
        init_image = self.resize_image(init_image, max_width, max_height)
        width, height = init_image.size
        width, height = self.closest_multiple_of_8(width, height)
        init_image = init_image.resize((width, height))

        mask_image = Image.open(mask).convert("L").resize((width, height))

        # Create inpainting condition
        inpainting_control_image = self.make_inpaint_condition(init_image, mask_image)

        # Create lineart condition
        lineart_control_image = self.lineart_processor(init_image)
        lineart_control_image = lineart_control_image.resize((width, height))

        # Pack control images for multiple ControlNets
        images = [inpainting_control_image, lineart_control_image]

        # Run the pipeline
        result = self.pipe(
            prompt_embeds=self.compel_proc(prompt),
            negative_prompt_embeds=self.compel_proc(negative_prompt),
            num_inference_steps=steps,
            generator=generator,
            eta=1,
            image=init_image,
            mask_image=mask_image,
            control_image=images,
            controlnet_conditioning_scale=strength,
            guidance_scale=guidance_scale,
        )

        out_image = result.images[0]
        out_image.save(out_path)
        return out_path
    
    
from deepface import DeepFace

def age_mapping(age: int) -> str:
    """
    Converts a numeric age to a rough age-group label.
    Customize the thresholds and labels as needed.
    """
    if age < 13:
        return "child"
    elif 13 <= age < 20:
        return "teen"
    elif 20 <= age < 30:
        return "young adult"
    elif 30 <= age < 50:
        return "middle-aged adult"
    else:
        return "elderly adult"

def create_prompt(deepface_result: dict) -> str:
    """
    Takes a single DeepFace analyze output (as a dictionary)
    and returns a descriptive prompt for Stable Diffusion.
    """
    
    deepface_result = deepface_result[0]
    
    age = deepface_result.get('age', 30)  # default 30 if missing
    dominant_gender = deepface_result.get('dominant_gender', 'Man').lower()
    dominant_race = deepface_result.get('dominant_race', 'white').lower()
    dominant_emotion = deepface_result.get('dominant_emotion', 'neutral').lower()
    
    age_group = age_mapping(age)
    race_descriptor = dominant_race.title()
    gender_descriptor = dominant_gender.lower()
    emotion_descriptor = "neutral"
    
    prompt = (
        f"A photorealistic portrait of a {age_group} {race_descriptor} {gender_descriptor}, "
        f"with a {emotion_descriptor} expression, looking directly at the camera. "
        "Ultra-detailed, 8k resolution, professional photography"
    )
    print(prompt)
    return prompt


if __name__ == "__main__":
    """
    Example CLI usage (very barebones):

    python predictor.py \
      --image /path/to/input.jpg \
      --mask /path/to/mask.png \
      --prompt "A tabby cat on a bench" \
      --negative_prompt "deformed, ugly" \
      --strength 0.8 \
      --max_height 612 \
      --max_width 612 \
      --steps 20 \
      --seed 42 \
      --guidance_scale 10
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask image")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--out_path", type=str, default="./res")
    args = parser.parse_args()
    
    im_name = os.path.splitext(os.path.basename(args.image))[0]
    args.out_path = os.path.join(args.out_path, f"{im_name}_out.png")
    
    objs = DeepFace.analyze(
        img_path = args.image, 
        actions = ['age', 'gender', 'race', 'emotion'],
        )
    
    if args.prompt == "":
        args.prompt = create_prompt(objs)
        
    if args.negative_prompt == "":
        args.negative_prompt = (
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, "
            "anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, "
            "bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, "
            "mutation, mutated, ugly, disgusting, amputation"
        )


    predictor = Predictor()
    output_path = predictor.predict(
        image=args.image,
        prompt=args.prompt,
        mask=args.mask,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        max_height=args.max_height,
        max_width=args.max_width,
        steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        out_path = args.out_path
    )

    print(f"Saved output image to: {output_path}")
