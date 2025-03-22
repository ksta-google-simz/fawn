import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image, make_image_grid
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
import face_alignment
from utils.anonymize_faces_in_image import anonymize_faces_in_image
from face_alignment import FaceAlignment, LandmarksType
from utils.face_embedding import save_exclusion_faces
import os

# 📌 2️⃣ 익명화 모델 로드
face_model_id = "hkung/face-anon-simple"
clip_model_id = "openai/clip-vit-large-patch14"
sd_model_id = "stabilityai/stable-diffusion-2-1"

unet = UNet2DConditionModel.from_pretrained(
    face_model_id, subfolder="unet", use_safetensors=True
)
referencenet = ReferenceNetModel.from_pretrained(
    face_model_id, subfolder="referencenet", use_safetensors=True
)
conditioning_referencenet = ReferenceNetModel.from_pretrained(
    face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
)
vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
scheduler = DDPMScheduler.from_pretrained(
    sd_model_id, subfolder="scheduler", use_safetensors=True
)
feature_extractor = CLIPImageProcessor.from_pretrained(
    clip_model_id, use_safetensors=True
)
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

# 📌 3️⃣ 디바이스 설정 (Mac 사용자는 MPS 활용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)

generator = torch.manual_seed(1)  # 랜덤성 고정하여 동일한 결과 출력
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 📌 4️⃣ 입력 이미지 로드 (특정 인물 제외 익명화 대상)
original_image = load_image("my_dataset/original/3.jpg")

# 📌 5️⃣ 얼굴 랜드마크 탐지기 초기화
fa = FaceAlignment(LandmarksType.TWO_D, device=device)

# 📌 6️⃣ 특정 인물 제외하고 익명화 수행
anon_image = anonymize_faces_in_image(
    image=original_image,
    face_alignment=fa,
    pipe=pipe,
    generator=generator,
    face_image_size=512,
    num_inference_steps=25,
    guidance_scale=4.0,
    anonymization_degree=1.25,  # 익명화 강도 조절 가능
)
output_path= ("my_dataset/anonymized/3_anon.jpg")
anon_image.save(output_path)
# 📌 7️⃣ 익명화 전후 비교 이미지 생성
# grid_image=make_image_grid([original_image, anon_image], rows=1, cols=2)
# grid_image.show()