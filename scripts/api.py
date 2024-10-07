import os
import random
import string
from cvt_utils import progress
import torch
from cvt_model.cloth_masker import AutoMasker as AM
from cvt_model.cloth_masker import vis_mask
from cvt_model.pipeline import CvtPipeline
from cvt_utils.utils import resize_and_crop, resize_and_padding
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
import gradio as gr
import numpy as np
from typing import Optional
from PIL import ImageFilter, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from fastapi import FastAPI
from pydantic import BaseModel
from modules.api import api
import threading
import time
from helper.logging import Logger
logger = Logger("API")

class CvtTryon:
    def load_pipeline(self, sd15_inpaint_path, cvt_path, mixed_precision):
        mixed_precision = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[mixed_precision]
        pipeline = CvtPipeline(
            base_ckpt=sd15_inpaint_path,
            attn_ckpt=cvt_path,
            attn_ckpt_version="mix",
            weight_dtype=mixed_precision,
            use_tf32=True,
            device='cuda'
        )
        return pipeline

    def load_mask_generator(self, cvt_path):
        cvt_path = snapshot_download(repo_id=cvt_path)
        automasker = AM(
            densepose_ckpt=os.path.join(cvt_path, "DensePose"),
            schp_ckpt=os.path.join(cvt_path, "SCHP"),
            device='cuda', 
        )
        return automasker


    def tryon_generate(
        self, 
        pipe: CvtPipeline, 
        target_image, 
        refer_image, 
        mask_image,
        seed = 42,
        steps = 50,
        cfg = 2.5,
        task_id=None,
    ):
        width, height = target_image.size
        generator = torch.Generator(device='cuda').manual_seed(seed)
        person_image = resize_and_crop(target_image, (width, height))
        cloth_image = resize_and_padding(refer_image, (width, height))
        mask = resize_and_crop(mask_image, (width, height))
        mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        mask = mask_processor.blur(mask, blur_factor=9)

        # Inference
        result_image = pipe(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            width=width,
            height=height,
            task_id=task_id,
        )[0]
        
        return result_image
    
    def auto_mask_generate(
        self, pipe, target_image, cloth_type
    ):
        width, height = target_image.size
        person_image = resize_and_crop(target_image, (width, height))
        mask = pipe(
            person_image,
            cloth_type
        )['mask']
        
        masked_image = vis_mask(person_image, mask)
        return (mask, masked_image)

class CvtMaskRequest(BaseModel):
    image: str
    mask_type: str = "overall"

class CvtTryOutfitRequest(BaseModel):
    cloth_image: str
    mask_image: Optional[str] = ""
    model_image: str
    mask_type: Optional[str] = "overall"
    params: Optional[dict] = {}

cvt_tryon = CvtTryon()
pipe = None
mask_generator = None

def cvt_load_pipeline(_):
    global pipe, mask_generator
    if pipe is None:
        print("Loading CvtPipeline")
        pipe = cvt_tryon.load_pipeline("annh/stable-diffusion-v1-5-inpainting", "annh/cvt", "fp16")
        print("CvtPipeline Loaded")
    if mask_generator is None:
        print("Loading AutoMask Generator")
        mask_generator = cvt_tryon.load_mask_generator("annh/cvt")
        print("AutoMask Generator Loaded")

def cvt_api(_: gr.Blocks, app: FastAPI):
    def cvtTryOutfit(
        data: CvtTryOutfitRequest,
        task_id=None
    ):
        try:
            progress.add_task(task_id)
            model_image = api.decode_base64_to_image(data.model_image)
            cloth_image = api.decode_base64_to_image(data.cloth_image)
            width, height = model_image.size
            new_width = width // 8 * 8
            new_height = height // 8 * 8
            model_image = model_image.resize((new_width, new_height))
            cloth_image = cloth_image.resize((new_width, new_height))

            if data.mask_image:
                mask_image = api.decode_base64_to_image(data.mask_image)
            else:
                if not data.mask_type:
                    return {
                        "error": "mask_type is required if mask_image is not provided"
                    }
                (mask_image, _) = cvt_tryon.auto_mask_generate(mask_generator, model_image, data.mask_type)
            progress.update_task(task_id, progress.CvtStatus.Active)
            result_image = cvt_tryon.tryon_generate(pipe, model_image, cloth_image, mask_image, task_id=task_id, **data.params)
            
            # paste cloth image to result image with mask inverted
            inverted_mask = ImageOps.invert(mask_image)
            blurred_mask = inverted_mask.filter(ImageFilter.GaussianBlur(4))
            result_image.paste(model_image, (0, 0), mask=blurred_mask)
            
            result = [
                api.encode_pil_to_base64(result_image),
                api.encode_pil_to_base64(mask_image),
            ]
            progress.update_task(task_id, progress.CvtStatus.Completed, 1.0, result=result)
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            progress.update_task(task_id, progress.CvtStatus.Failed, result=str(e))
            raise e
        
    @app.post("/sdapi/v2/cvt/getmask")
    def cvtGetMask(
        data: CvtMaskRequest
    ):
        image = api.decode_base64_to_image(data.image)
        mask, masked_image = cvt_tryon.auto_mask_generate(mask_generator, image, data.mask_type)
        return [
            api.encode_pil_to_base64(mask),
            api.encode_pil_to_base64(masked_image),
        ]

        
    @app.post("/sdapi/v2/cvt/try-outfit-sync")
    def cvtTryOutfitAPI(
        data: CvtTryOutfitRequest
    ):
        return cvtTryOutfit(data)
        
    @app.post("/sdapi/v2/cvt/try-outfit")
    def cvtTryOutfitAPI2(
        data: CvtTryOutfitRequest
    ):
        logger.info(f"===== API /sdapi/v2/cvt/try-outfit =====")
        task_id = ''.join(random.choice(string.ascii_letters) for i in range(10))
        thread = threading.Thread(target=cvtTryOutfit, args=(data,task_id))
        thread.start()
        return {
            "status" : "success",
            "task_id" : task_id,
        }
    
    @app.get("/sdapi/v2/cvt/task/{task_id}")
    def cvtGetTask(
        task_id: str
    ):
        task = progress.get_task(task_id)
        return task

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(cvt_api)
    script_callbacks.on_model_loaded(cvt_load_pipeline)
except:
    pass
