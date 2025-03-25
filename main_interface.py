# frontend.py (수정된 run_pipeline)
from PIL import Image
import numpy as np
import tempfile
import os
import shutil

from single_anon import get_single_anon
from utils.face_embedding import save_exclusion_faces

def run_pipeline(image_path, exclusion_image=None):
    # 1. 익명화 제외 얼굴 등록 (옵션)
    if exclusion_image is not None:
        # 💡 ndarray -> RGB 이미지로 변환
        if isinstance(exclusion_image, np.ndarray):
            if exclusion_image.dtype != np.uint8:
                exclusion_image = (exclusion_image * 255).astype(np.uint8)

            if exclusion_image.shape[-1] == 4:  # RGBA일 경우
                exclusion_image = exclusion_image[:, :, :3]

            exclusion_image = Image.fromarray(exclusion_image).convert("RGB")
        else:
            exclusion_image = Image.open(exclusion_image).convert("RGB")

        temp_dir = tempfile.gettempdir()
        exclusion_path = os.path.join(temp_dir, "exclusion_rgb.jpg")
        exclusion_image.save(exclusion_path)

        save_exclusion_faces([exclusion_path])

    # 2. 입력 이미지 저장
    input_dir = "my_dataset/run/"
    output_dir = "my_dataset/anonymized/"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_image_path = os.path.join(input_dir, "input.jpg")
    shutil.copy(image_path, input_image_path)

    # 3. 익명화 수행
    get_single_anon(input_dir, output_dir, num_inference_steps=10, anonymization_degree=1.25)

    # 4. 출력 이미지 경로
    output_image_path = os.path.join(output_dir, "input.jpg")
    return output_image_path
