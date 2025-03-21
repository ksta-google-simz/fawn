import os
import pickle
import face_recognition
from PIL import Image
import numpy as np

# ✅ 지원하는 이미지 확장자 목록
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def is_valid_image(filename):
    """유효한 이미지 파일인지 검사"""
    return filename.lower().endswith(tuple(IMAGE_EXTENSIONS))

# ✅ 데이터 경로 설정
original_path = "my_dataset/original/"
anonymized_path = "my_dataset/anonymized/"
dataset_path = "my_dataset/dataset_faces/"
failed_image_path = "failed_images/"  # 감지 실패한 이미지 저장 폴더

# ✅ 감지 실패한 이미지를 저장할 폴더 생성
os.makedirs(failed_image_path, exist_ok=True)

# ✅ 임베딩 저장 딕셔너리
original_embeddings = {}
anonymized_embeddings = {}
dataset_embeddings = {}

# ✅ 로드 실패한 이미지 및 얼굴 감지 실패 이미지 저장 리스트
failed_images = []
no_face_detected = []

def process_images(folder_path, embedding_dict, category):
    """ 이미지 폴더를 순회하며 face_recognition을 사용하여 얼굴 벡터 추출 """
    image_count = 0
    total_images = [f for f in os.listdir(folder_path) if is_valid_image(f)]
    
    print(f"\n📂 {category} 이미지 파일 개수: {len(total_images)}")

    for filename in total_images:
        img_path = os.path.join(folder_path, filename)

        try:
            img = face_recognition.load_image_file(img_path)  # ✅ `face_recognition`으로 이미지 로드
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {filename}, 오류: {e}")
            failed_images.append(img_path)
            continue  # 이미지 로드 실패 시 건너뜀

        face_encodings = face_recognition.face_encodings(img)  # 🔥 얼굴 임베딩 생성

        if face_encodings:
            print(f"🚀 {filename}에서 감지된 얼굴 수: {len(face_encodings)}")  # ✅ 감지된 얼굴 개수 출력
            embedding_dict[filename] = face_encodings[0]  # ✅ 첫 번째 얼굴 벡터 저장
            image_count += 1
        else:
            print(f"⚠️ {filename}에서 얼굴 감지 실패")  
            no_face_detected.append(img_path)
            Image.open(img_path).save(os.path.join(failed_image_path, filename))  # ❌ 감지 실패한 이미지 저장

    return image_count

# ✅ 원본 얼굴 벡터 저장
original_image_count = process_images(original_path, original_embeddings, "원본")
# ✅ 익명화된 얼굴 벡터 저장
anonymized_image_count = process_images(anonymized_path, anonymized_embeddings, "익명화된")
# ✅ 데이터셋 얼굴 벡터 저장 (1:N, Rank-K 평가용)
dataset_image_count = process_images(dataset_path, dataset_embeddings, "비교 데이터셋")

# ✅ 실패한 이미지 파일 출력
if failed_images:
    print("\n🚨 로드 실패한 이미지 목록:")
    for failed_img in failed_images:
        print(f"❌ {failed_img}")

if no_face_detected:
    print("\n⚠️ 얼굴 감지 실패한 이미지 목록 (감지 실패 이미지 저장됨):")
    for img in no_face_detected:
        print(f"⚠️ {img}")

# ✅ 최종 통계 출력
print("\n📌 최종 이미지 데이터 개수")
print(f"✅ 원본 폴더 내 이미지 개수: {original_image_count}")
print(f"✅ 익명화된 폴더 내 이미지 개수: {anonymized_image_count}")
print(f"✅ 비교 데이터셋 폴더 내 이미지 개수: {dataset_image_count}")

# 🔹 벡터 파일 저장 (pickle 사용)
with open("original_embeddings.pkl", "wb") as f:
    pickle.dump(original_embeddings, f)

with open("anonymized_embeddings.pkl", "wb") as f:
    pickle.dump(anonymized_embeddings, f)

with open("dataset_embeddings.pkl", "wb") as f:
    pickle.dump(dataset_embeddings, f)

print("✅ 얼굴 벡터 저장 완료!")