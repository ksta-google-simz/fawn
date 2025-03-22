import os
from diffusers.utils import load_image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
import face_alignment
from utils.anonymize_faces_in_image import anonymize_faces_in_image
import sys

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


pipe = pipe.to("cuda")

generator = torch.manual_seed(1)  # 랜덤성 고정하여 동일한 결과 출력


def get_single_anon(input_folder, output_folder, num_inference_steps, anonymization_degree):
    # 익명화할 이미지가 저장된 폴더 경로
    # input_folder = "my_dataset/original/"
    # output_folder = "my_dataset/anonymized/"

    # 입력 폴더가 없으면 경고 후 종료
    if not os.path.exists(input_folder):
        print(f"경고: ]]입력 폴더 '{input_folder}'가 존재하지 않습니다.", file=sys.stderr)
        sys.exit(1)

    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 폴더 내 모든 이미지 파일 가져오기
    image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

    # 이미지 익명화 진행
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)  # 원본 이미지 경로
        # 파일명과 확장자 분리
        filename, ext = os.path.splitext(image_file)
        
        # 익명화된 이미지 저장 경로 (원본파일명_anon.확장자)
        output_path = os.path.join(output_folder, f"{filename}{ext}")
        
        print(f"🔄 익명화 진행: {image_file}")

        # 원본 이미지 로드
        original_image = load_image(input_path)
        # SFD (likely best results, but slower)

        fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
        )
        # generate an image that anonymizes faces
        anon_image = anonymize_faces_in_image(
            image=original_image,
            face_alignment=fa,
            pipe=pipe,
            generator=generator,
            face_image_size=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=4.0,
            anonymization_degree=anonymization_degree,
        )
        # 이미지 저장
        anon_image.save(output_path)
        print(f"✅ 저장 완료: {output_path}")


    print("🎉 모든 이미지 익명화 완료!")
