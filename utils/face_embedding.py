import face_recognition
import numpy as np
import pickle
import os
from PIL import Image

EXCLUSION_FACE_IMAGES_DIR = "exclusion_faces_imgs"  # ✅ 새 폴더 경로 추가
EXCLUSION_FACE_FEATURES_FILE = "exclusion_faces.pkl"  # 저장할 파일

def save_exclusion_faces(image_paths):
    """여러 명의 얼굴 특징을 저장하고 썸네일도 저장"""
    exclusion_faces = []

    # 기존 얼굴 로드
    if os.path.exists(EXCLUSION_FACE_FEATURES_FILE):
        with open(EXCLUSION_FACE_FEATURES_FILE, "rb") as f:
            exclusion_faces = pickle.load(f)

    os.makedirs(EXCLUSION_FACE_IMAGES_DIR, exist_ok=True)

    start_index = len(exclusion_faces)  # ✅ 현재까지 저장된 얼굴 개수

    for i, image_path in enumerate(image_paths):
        try:
            pil_image = Image.open(image_path).convert("RGB")
            image = np.array(pil_image)

            face_encodings = face_recognition.face_encodings(image)
            print(f"🔍 {image_path}에서 감지된 얼굴 수: {len(face_encodings)}")

            if len(face_encodings) > 0:
                exclusion_faces.append(face_encodings[0])
                print(f"✅ {image_path} 얼굴 저장 완료")

                # ✅ 썸네일 저장 (덮어쓰기 방지)
                index = start_index + i + 1
                thumb_path = os.path.join(EXCLUSION_FACE_IMAGES_DIR, f"face_{index:03}.jpg")
                pil_image.save(thumb_path)
            else:
                print(f"❌ {image_path} 얼굴을 찾을 수 없습니다.")
        except Exception as e:
            print(f"❌ {image_path} 처리 중 오류 발생: {e}")

    with open(EXCLUSION_FACE_FEATURES_FILE, "wb") as f:
        pickle.dump(exclusion_faces, f)

    print(f"✅ 총 {len(exclusion_faces)}명의 얼굴이 익명화 제외 대상으로 저장되었습니다.")


def load_exclusion_faces():
    """저장된 얼굴 특징을 불러오는 함수"""
    try:
        with open(EXCLUSION_FACE_FEATURES_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []


def reset_exclusion_faces():
    """익명화 제외 대상 초기화"""
    if os.path.exists(EXCLUSION_FACE_FEATURES_FILE):
        os.remove(EXCLUSION_FACE_FEATURES_FILE)
        print("✅ 익명화 제외 대상이 초기화되었습니다.")
    else:
        print("⚠️ 익명화 제외 대상이 존재하지 않습니다.")

    # ✅ 썸네일 폴더도 초기화
    if os.path.exists(EXCLUSION_FACE_IMAGES_DIR):
        for f in os.listdir(EXCLUSION_FACE_IMAGES_DIR):
            os.remove(os.path.join(EXCLUSION_FACE_IMAGES_DIR, f))
        print("🧹 썸네일 이미지도 삭제 완료")


def is_exclusion_face(face_image, euclidean_threshold=0.4):
    """감지된 얼굴이 익명화 제외 대상인지 비교하는 함수"""
    exclusion_faces = load_exclusion_faces()
    if len(exclusion_faces) == 0:
        return False  # 저장된 얼굴 없음

    try:
        # PIL → RGB → np.array
        if isinstance(face_image, Image.Image):
            face_image = face_image.convert("RGB")
            face_image = np.array(face_image)

        face_encodings = face_recognition.face_encodings(face_image)
        if len(face_encodings) == 0:
            return False

        face_vector = face_encodings[0]
        for exclusion_face in exclusion_faces:
            distance = face_recognition.face_distance([exclusion_face], face_vector)[0]
            print(f"🔎 얼굴 비교 - 유클리드 거리: {distance:.3f}")
            if distance < euclidean_threshold:
                return True  # 익명화 제외 대상
    except Exception as e:
        print(f"❌ is_exclusion_face 오류: {e}")
        return False

    return False  # 익명화 대상

def load_exclusion_face_images():
    """등록된 얼굴 이미지들 불러오기"""
    if not os.path.exists(EXCLUSION_FACE_IMAGES_DIR):
        return []
    files = sorted(os.listdir(EXCLUSION_FACE_IMAGES_DIR))
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # 원하는 확장자 추가 가능
    paths = [os.path.join(EXCLUSION_FACE_IMAGES_DIR, f) for f in files if f.lower().endswith(valid_exts)]
    return paths
