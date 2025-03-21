import face_alignment
from PIL import Image
import torch

from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
from utils.extractor import extract_faces
from utils.merger import paste_foreground_onto_background
from utils.face_embedding import is_exclusion_face


# def anonymize_faces_in_image(
#     image: Image,
#     face_alignment: face_alignment.FaceAlignment,
#     pipe: StableDiffusionReferenceNetPipeline,
#     generator: torch.Generator = None,
#     face_image_size: int = 512,
#     num_inference_steps: int = 50,
#     guidance_scale: float = 4,
#     anonymization_degree: float = 1.25,
# ) -> Image:
#     face_images, image_to_face_matrices = extract_faces(
#         face_alignment, image, face_image_size
#     )

#     anon_image = image
#     for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
#         if is_exclusion_face(face_image):
#             print("✅ 특정 인물 감지됨: 익명화 제외")
#             continue  # 익명화 대상에서 제외하고 원본 얼굴 유지
#         # generate an image that anonymizes faces
#         anon_face_image = pipe(
#             source_image=face_image,
#             conditioning_image=face_image,
#             num_inference_steps=num_inference_steps,
#             guidance_scale=guidance_scale,
#             generator=generator,
#             anonymization_degree=anonymization_degree,
#         ).images[0]

#         anon_image = paste_foreground_onto_background(
#             anon_face_image, anon_image, image_to_face_mat
#         )

#     return anon_image

def anonymize_faces_in_image(
    image: Image,
    face_alignment: face_alignment.FaceAlignment,
    pipe: StableDiffusionReferenceNetPipeline,
    generator: torch.Generator = None,
    face_image_size: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 4,
    anonymization_degree: float = 1.25,
) -> Image:
    face_images, image_to_face_matrices = extract_faces(
        face_alignment, image, face_image_size
    )  # 이미 익명화 제외 대상은 제거됨

    anon_image = image  # 원본 이미지 복사

    for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
        # generate an image that anonymizes faces
        anon_face_image = pipe(
            source_image=face_image,
            conditioning_image=face_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            anonymization_degree=anonymization_degree,
        ).images[0]

        # 익명화된 얼굴을 원본 이미지에 합성
        anon_image = paste_foreground_onto_background(
            anon_face_image, anon_image, image_to_face_mat
        )

    return anon_image
