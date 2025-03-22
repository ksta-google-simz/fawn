import argparse
from single_anon import get_single_anon
from utils.eval import eval_fid


def main():
    parser = argparse.ArgumentParser(description="이미지 익명화 및 평가 스크립트")
    parser.add_argument("--origin_path", type=str, required=True, help="원본 이미지 폴더 경로")
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

    get_single_anon(args.origin_path, args.anon_path, args.num_inference_steps, args.anonymization_degree)
    print("#" * 100)
    eval_fid(args.origin_path, args.anon_path)
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
