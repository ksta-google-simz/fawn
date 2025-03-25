import argparse
import pickle
import time
from single_anon import get_single_anon
from utils.eval import eval_fid, process_images_to_pickle, face_identification_reid, face_similarity


def main():
    parser = argparse.ArgumentParser(description="이미지 익명화 및 평가 스크립트")
    parser.add_argument("--origin_path", type=str, default="my_dataset/original/", help="원본 이미지 폴더 경로")
    parser.add_argument("--anon_path", type=str, default="my_dataset/anon/", help="익명화 이미지 저장 폴더 경로")
    parser.add_argument("--label_path", type=str, default="my_dataset/fairface_label_val.csv", help="agr 관련 수치들 저장 경로")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="추론 단계 수 (default: 25)")
    parser.add_argument("--anonymization_degree", type=float, default=1.25, help="익명화 정도 (default: 1.25)")
    
    args = parser.parse_args()
    print("#" * 50)
    print("#  ✅ 입력 받은 인자:")
    for arg, value in vars(args).items():
        print(f"#  📌 {arg}: {value}")

    print("#" * 50)

    start_time = time.time()
    get_single_anon(args.origin_path, args.anon_path, args.num_inference_steps, args.anonymization_degree)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"🔹 익명화에 걸린 시간: {elapsed_time:.2f}초")

    print("#" * 100)
    
    eval_fid(args.origin_path, args.anon_path)

    print("#" * 100)
    
    # process_images_to_pickle(args.origin_path, {}, 'original')
    process_images_to_pickle(args.anon_path, {}, 'anonymized')

    # ✅ 저장된 얼굴 벡터 로드
    with open("original_embeddings.pkl", "rb") as f:
        original_embeddings = pickle.load(f)

    with open("anonymized_embeddings.pkl", "rb") as f:
        anonymized_embeddings = pickle.load(f)

    with open("dataset_embeddings.pkl", "rb") as f:
        dataset_embeddings = pickle.load(f)

    verification_reid, avg_sim = face_similarity(original_embeddings, anonymized_embeddings, threshold=0.8)
    print(f"🔹 Face Identification (1:1) Re-ID Rate: {verification_reid * 100:.2f}%")
    print(f"🔹 Face Average Similarity : {avg_sim * 100:.2f}%")

    identification_score1, identification_score2 = face_identification_reid(original_embeddings, anonymized_embeddings, dataset_embeddings)
    
    # ✅ 결과 출력
    print(f"🔹 Face Identification (1:N) Re-ID Rate: {identification_score1 * 100:.2f}% (denominator = num of original = 200)")
    print(f"🔹 Face Identification (1:N) Re-ID Rate: {identification_score2 * 100:.2f}% (denominator = num of anonymized)")

    # print("#" * 100)
    # orig_stats, anon_stats, diff_stats, diff_df = compare_agr(args.origin_path, args.anon_path, args.label_path)
    # print("\nOriginal image statistics:")
    # for key, value in orig_stats.items():
    #     print(f"{key}: {value:.4f}")
    # print("\nAnonymized image statistics:")
    # for key, value in anon_stats.items():
    #     print(f"{key}: {value:.4f}")
    # print("\nDifference statistics:")
    # for key, value in diff_stats.items():
    #     print(f"{key}: {value:.4f}")
    # print("\nFirst few comparison results:")
    # print(diff_df.head())

if __name__ == "__main__":
    main()
