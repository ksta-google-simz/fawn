import os
import shutil
from pytorch_fid import fid_score

def eval_fid(original_dir, anonymized_dir):
    temp_dir = "./temp_eval"
    os.makedirs(temp_dir, exist_ok=True)

    fids = []

    # 원본 이미지 기준으로 매칭
    for file in os.listdir(original_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filename, ext = os.path.splitext(file)
        anon_file = f"{filename}_anon{ext}"
        original_path = os.path.join(original_dir, file)
        anonymized_path = os.path.join(anonymized_dir, anon_file)

        if not os.path.exists(anonymized_path):
            print(f"⚠️ 익명화된 이미지 없음: {anon_file}")
            continue

        # 임시 폴더 초기화
        original_temp = os.path.join(temp_dir, "original")
        anonymized_temp = os.path.join(temp_dir, "anonymized")
        os.makedirs(original_temp, exist_ok=True)
        os.makedirs(anonymized_temp, exist_ok=True)

        shutil.copy(original_path, os.path.join(original_temp, "img.png"))
        shutil.copy(anonymized_path, os.path.join(anonymized_temp, "img.png"))

        fid_value = fid_score.calculate_fid_given_paths(
            [original_temp, anonymized_temp],
            batch_size=1, device="cuda", dims=2048
        )

        print(f"📊 {file} FID: {fid_value:.4f}")
        fids.append(fid_value)

        shutil.rmtree(original_temp)
        shutil.rmtree(anonymized_temp)

    shutil.rmtree(temp_dir)

    if fids:
        avg_fid = sum(fids) / len(fids)
        print(f"\n📈 평균 FID: {avg_fid:.4f}")
        return fids, avg_fid
    else:
        print("⚠️ 평가 가능한 이미지 쌍이 없습니다.")
        return [], None
