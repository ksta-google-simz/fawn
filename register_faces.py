from utils.face_embedding import save_exclusion_faces

# 📌 익명화 제외할 얼굴 등록
image_paths = ["karina.png"]  # 등록할 얼굴 이미지 파일들
save_exclusion_faces(image_paths)