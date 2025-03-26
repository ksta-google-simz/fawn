# model_setup.py
import torch
import face_alignment
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline

# 모델 경로 설정
face_model_id = "hkung/face-anon-simple"
clip_model_id = "openai/clip-vit-large-patch14"
sd_model_id = "stabilityai/stable-diffusion-2-1"

# 모델 로드
unet = UNet2DConditionModel.from_pretrained(face_model_id, subfolder="unet", use_safetensors=True)
referencenet = ReferenceNetModel.from_pretrained(face_model_id, subfolder="referencenet", use_safetensors=True)
conditioning_referencenet = ReferenceNetModel.from_pretrained(face_model_id, subfolder="conditioning_referencenet", use_safetensors=True)
vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
scheduler = DDPMScheduler.from_pretrained(sd_model_id, subfolder="scheduler", use_safetensors=True)
feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id, use_safetensors=True)
image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

pipe = StableDiffusionReferenceNetPipeline(
    unet=unet,
    referencenet=referencenet,
    conditioning_referencenet=conditioning_referencenet,
    vae=vae,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    scheduler=scheduler,
)

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
generator = torch.manual_seed(1)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)