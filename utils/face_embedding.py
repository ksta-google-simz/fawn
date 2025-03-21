import face_recognition
import numpy as np
import pickle
import os

EXCLUSION_FACE_FEATURES_FILE = "exclusion_faces.pkl"  # 저장할 파일

def save_exclusion_faces(image_paths):
    """ 여러 명의 얼굴 특징을 저장하는 함수 """
    exclusion_faces = []  # 기존 얼굴 리스트

    # 기존 등록된 얼굴 불러오기 (있으면 추가)
    if os.path.exists(EXCLUSION_FACE_FEATURES_FILE):
        with open(EXCLUSION_FACE_FEATURES_FILE, "rb") as f:
            exclusion_faces = pickle.load(f)

    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            exclusion_faces.append(face_encodings[0])  # 첫 번째 얼굴 특징 저장
            print(f"✅ {image_path} 얼굴 저장 완료")
        else:
            print(f"❌ {image_path} 얼굴을 찾을 수 없습니다.")
    
    # 업데이트된 얼굴 리스트 저장
    with open(EXCLUSION_FACE_FEATURES_FILE, "wb") as f:
        pickle.dump(exclusion_faces, f)
    print(f"✅ 총 {len(exclusion_faces)}명의 얼굴이 익명화 제외 대상으로 저장되었습니다.")


def load_exclusion_faces():
    """ 저장된 얼굴 특징을 불러오는 함수 """
    try:
        with open(EXCLUSION_FACE_FEATURES_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []
    

def reset_exclusion_faces():
    """ 익명화 제외 대상 초기화 """
    if os.path.exists(EXCLUSION_FACE_FEATURES_FILE):
        try:
            os.remove(EXCLUSION_FACE_FEATURES_FILE)
            print("✅ 익명화 제외 대상이 초기화되었습니다.")
        except Exception as e:
            print(f"❌ 파일 삭제 중 오류 발생: {e}")
    else:
        print("⚠️ 익명화 제외 대상이 존재하지 않습니다.")



def is_exclusion_face(face_image, euclidean_threshold=0.4):
    """ 감지된 얼굴이 익명화 제외 대상인지 비교하는 함수 (유클리드 거리만 사용) """
    exclusion_faces = load_exclusion_faces()
    if len(exclusion_faces) == 0:
        return False  # 저장된 얼굴이 없으면 익명화 진행

    face_encodings = face_recognition.face_encodings(np.array(face_image))
    
    if len(face_encodings) > 0:
        face_vector = face_encodings[0]
        for exclusion_face in exclusion_faces:
            # 🔹 유클리드 거리 계산
            distance = face_recognition.face_distance([exclusion_face], face_vector)[0]

            print(f"🔎 얼굴 비교 - 유클리드 거리: {distance:.3f}")  # 디버깅용 출력

            # 유클리드 거리 기준 충족하면 익명화 제외
            if distance < euclidean_threshold:
                return True  # 익명화 제외 대상

    return False  # 익명화 진행
