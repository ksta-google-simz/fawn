import pickle
import numpy as np
from scipy.spatial.distance import cosine

# ✅ 1:1 얼굴 검증 방식 (Face Verification)
def face_verification_reid(original_embeddings, anonymized_embeddings, threshold=0.8):
    match_count = 0
    total_faces = len(original_embeddings)

    for filename, orig_emb in original_embeddings.items():
        if filename in anonymized_embeddings:
            anon_emb = anonymized_embeddings[filename]
            similarity = 1 - cosine(orig_emb, anon_emb)

            print(f"🔹 {filename} | 유사도: {similarity:.3f}")  # ✅ 유사도 출력
            
            if similarity > threshold:  # 임계값 이상이면 동일 인물로 판단
                match_count += 1

    return match_count / total_faces if total_faces > 0 else 0

# ✅ 1:N 얼굴 식별 (Face Identification)
def face_identification_reid(original_embeddings, anonymized_embeddings, dataset_embeddings):
    match_count = 0
    total_faces = len(original_embeddings)

    for filename, orig_emb in original_embeddings.items():
        if filename in anonymized_embeddings:
            anon_emb = anonymized_embeddings[filename]

            similarities = np.array([
                1 - cosine(anon_emb, dataset_emb)
                for dataset_emb in dataset_embeddings.values()
            ])

            best_match_index = np.argmax(similarities)
            best_match_filename = list(dataset_embeddings.keys())[best_match_index]
            best_match_score = similarities[best_match_index]

            print(f"🔍 {filename} | 최고 유사도 얼굴: {best_match_filename} | 유사도: {best_match_score:.3f}")

            if best_match_filename == filename:
                match_count += 1

    return match_count / total_faces if total_faces > 0 else 0

# ✅ Rank-K 평가
def rank_k_reid(original_embeddings, anonymized_embeddings, dataset_embeddings, k=5):
    match_count = 0
    total_faces = len(original_embeddings)

    for filename, orig_emb in original_embeddings.items():
        if filename in anonymized_embeddings:
            anon_emb = anonymized_embeddings[filename]

            similarities = np.array([
                1 - cosine(anon_emb, dataset_emb)
                for dataset_emb in dataset_embeddings.values()
            ])

            top_k_indices = np.argsort(similarities)[-k:]  # 유사도 상위 K개 인덱스

            if list(dataset_embeddings.keys()).index(filename) in top_k_indices:
                match_count += 1

    return match_count / total_faces if total_faces > 0 else 0

# ✅ 저장된 얼굴 벡터 로드
with open("original_embeddings.pkl", "rb") as f:
    original_embeddings = pickle.load(f)

with open("anonymized_embeddings.pkl", "rb") as f:
    anonymized_embeddings = pickle.load(f)

with open("dataset_embeddings.pkl", "rb") as f:
    dataset_embeddings = pickle.load(f)

print(f"✅ 원본 얼굴 개수: {len(original_embeddings)}")
print(f"✅ 익명화된 얼굴 개수: {len(anonymized_embeddings)}")
print(f"✅ 비교 데이터셋 얼굴 개수: {len(dataset_embeddings)}")

# ✅ Re-ID Rate 평가 실행
verification_score = face_verification_reid(original_embeddings, anonymized_embeddings, threshold=0.5)
identification_score = face_identification_reid(original_embeddings, anonymized_embeddings, dataset_embeddings)
rank_5_score = rank_k_reid(original_embeddings, anonymized_embeddings, dataset_embeddings, k=5)

# ✅ 결과 출력
print(f"🔹 Face Verification Re-ID Rate: {verification_score * 100:.2f}%")
print(f"🔹 Face Identification (1:N) Re-ID Rate: {identification_score * 100:.2f}%")
print(f"🔹 Rank-5 Re-ID Rate: {rank_5_score * 100:.2f}%")