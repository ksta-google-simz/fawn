import shutil
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pytorch_fid import fid_score
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine

import pickle
import face_recognition
from PIL import Image

def eval_fid(original_dir, anonymized_dir):
    temp_dir = "./temp_eval"
    os.makedirs(temp_dir, exist_ok=True)

    fids = []

    # 원본 이미지 기준으로 매칭
    for file in os.listdir(original_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filename, ext = os.path.splitext(file)
        anon_file = f"{filename}{ext}"
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


race_mapping = {
    'White': 'white',
    'Black': 'black',
    'Latino_Hispanic': 'latino hispanic',
    'East Asian': 'asian',
    'Southeast Asian': 'asian',
    'Indian': 'indian',
    'Middle Eastern': 'middle eastern'
}

def convert_age_to_number(age_str):
    age_mapping = {
        '0-2': 1,
        '3-9': 6,
        '10-19': 15,
        '20-29': 25,
        '30-39': 35,
        '40-49': 45,
        '50-59': 55,
        '60-69': 65,
        'more than 70': 75
    }
    return age_mapping[age_str]

def convert_gender(gender_str):
    gender_mapping = {
        'Woman': 'Female',
        'Man': 'Male'
    }
    return gender_mapping.get(gender_str, gender_str)

def compute_gender_cross_entropy(prediction, true_label):
    try:
        # 예측 확률 추출 (Woman 확률을 사용)
        gender_dict = prediction.get('gender', {})
        prob_female = gender_dict.get('Woman', 0.0) / 100.0  # 100으로 나누어 0~1 범위로 변환
        
        # 실제 라벨을 이진값으로 변환 (Female=1, Male=0)
        true_value = 1.0 if true_label.lower() == 'female' else 0.0
        
        # Binary Cross Entropy 계산
        epsilon = 1e-15
        prob_female = np.clip(prob_female, epsilon, 1 - epsilon)
        cross_entropy = -(true_value * np.log(prob_female) + (1 - true_value) * np.log(1 - prob_female))
        
        return cross_entropy
    except Exception as e:
        print(f"Error in gender cross entropy calculation: {str(e)}")
        return 0.0

def compute_race_cross_entropy(prediction, true_label):
    try:
        # 실제 라벨을 매핑된 값으로 변환
        true_race = race_mapping.get(true_label, true_label.lower())
        
        # 각 클래스의 예측 확률을 추출하고 정규화
        race_probs = prediction.get('race', {})
        if not race_probs:
            return 0.0
            
        total_prob = sum(race_probs.values())
        if total_prob == 0:
            return 0.0
        
        # 각 클래스에 대한 Cross Entropy 계산
        epsilon = 1e-15
        cross_entropy = 0
        
        for race, prob in race_probs.items():
            normalized_prob = prob / total_prob
            normalized_prob = np.clip(normalized_prob, epsilon, 1 - epsilon)
            true_value = 1.0 if race == true_race else 0.0
            cross_entropy -= true_value * np.log(normalized_prob)
        
        return cross_entropy
    except Exception as e:
        print(f"Error in race cross entropy calculation: {str(e)}")
        return 0.0

# ▶ 여러 얼굴 혹은 단 하나의 얼굴 검출 시 모두 대응하기 위한 보조 함수
def to_single_result(pred):
    """
    DeepFace.analyze() 결과가 list 또는 dict일 수 있으므로
    list면 첫 번째 요소, dict면 그대로 반환.
    """
    if isinstance(pred, list):
        if len(pred) > 0:
            return pred[0]
        else:
            return None
    elif isinstance(pred, dict):
        return pred
    else:
        return None

def compare_agr(original_dir, anonymized_dir, label_path, save_csv=True):
    labels_df = pd.read_csv(label_path)
    
    # 파일 이름 형식 변환 (예: val/5001.jpg -> 5001_anon.jpg)
    labels_df['file'] = labels_df['file'].apply(lambda x: os.path.basename(x).split('.')[0] + '_anon.jpg')
    
    differences = []
    total_predictions = 0
    
    image_files = glob.glob(os.path.join(anonymized_dir, "*.jpg")) + \
                  glob.glob(os.path.join(anonymized_dir, "*.png"))
    
    orig_metrics = {
        'age_errors': [],
        'gender_correct': 0,
        'gender_losses': [],
        'race_correct': 0,
        'race_losses': []
    }
    anon_metrics = {
        'age_errors': [],
        'gender_correct': 0,
        'gender_losses': [],
        'race_correct': 0,
        'race_losses': []
    }
    
    for anon_path in tqdm(image_files):
        try:
            img_id = os.path.basename(anon_path)
            orig_path = os.path.join(original_dir, img_id.replace('_anon', ''))
            
            # 해당 이미지의 라벨 찾기
            img_label = labels_df[labels_df['file'] == img_id].iloc[0]
            
            # 원본 이미지 분석
            orig_pred_raw = DeepFace.analyze(
                orig_path, actions=['age', 'gender', 'race'], 
                align=True, detector_backend='mtcnn'
            )
            # 익명화 이미지 분석
            anon_pred_raw = DeepFace.analyze(
                anon_path, actions=['age', 'gender', 'race'], 
                align=True, detector_backend='mtcnn'
            )
            
            # 리스트/딕셔너리 상태에 따라 단일 결과만 추출
            orig_pred = to_single_result(orig_pred_raw)
            anon_pred = to_single_result(anon_pred_raw)

            if (orig_pred is None) or (anon_pred is None):
                print(f"Invalid prediction format for {img_id}")
                continue
            
            # 결과 딕셔너리 구성
            result = {
                'file_name': img_id,
                'true_age_group': img_label['age'],
                'orig_age': orig_pred.get('age', 0),
                'anon_age': anon_pred.get('age', 0),
                'true_gender': img_label['gender'],
                'orig_gender': convert_gender(orig_pred.get('dominant_gender', '')),
                'anon_gender': convert_gender(anon_pred.get('dominant_gender', '')),
                'true_race': img_label['race'],
                'orig_race': orig_pred.get('dominant_race', '').lower(),
                'anon_race': anon_pred.get('dominant_race', '').lower()
            }
            
            # 나이 에러 계산
            try:
                true_age = convert_age_to_number(img_label['age'])
                orig_age_error = abs(true_age - orig_pred.get('age', 0))
                anon_age_error = abs(true_age - anon_pred.get('age', 0))
                result['age_error_diff'] = anon_age_error - orig_age_error
                result['orig_age_match'] = orig_age_error <= 5
                result['anon_age_match'] = anon_age_error <= 5
                
                orig_metrics['age_errors'].append(orig_age_error)
                anon_metrics['age_errors'].append(anon_age_error)
            except Exception as e:
                print(f"Error in age calculation for {img_id}: {str(e)}")
            
            # 성별 비교
            try:
                orig_gender = convert_gender(orig_pred.get('dominant_gender', ''))
                anon_gender = convert_gender(anon_pred.get('dominant_gender', ''))
                result['orig_gender_match'] = (orig_gender.lower() == img_label['gender'].lower())
                result['anon_gender_match'] = (anon_gender.lower() == img_label['gender'].lower())
                
                if result['orig_gender_match']:
                    orig_metrics['gender_correct'] += 1
                if result['anon_gender_match']:
                    anon_metrics['gender_correct'] += 1
                
                orig_gender_ce = compute_gender_cross_entropy(orig_pred, img_label['gender'])
                anon_gender_ce = compute_gender_cross_entropy(anon_pred, img_label['gender'])
                result['gender_ce_diff'] = anon_gender_ce - orig_gender_ce
                
                orig_metrics['gender_losses'].append(orig_gender_ce)
                anon_metrics['gender_losses'].append(anon_gender_ce)
            except Exception as e:
                print(f"Error in gender calculation for {img_id}: {str(e)}")
            
            # 인종 비교
            try:
                mapped_true_race = race_mapping.get(img_label['race'], img_label['race'].lower())
                result['orig_race_match'] = (orig_pred.get('dominant_race', '').lower() == mapped_true_race)
                result['anon_race_match'] = (anon_pred.get('dominant_race', '').lower() == mapped_true_race)
                
                if result['orig_race_match']:
                    orig_metrics['race_correct'] += 1
                if result['anon_race_match']:
                    anon_metrics['race_correct'] += 1
                
                orig_race_ce = compute_race_cross_entropy(orig_pred, img_label['race'])
                anon_race_ce = compute_race_cross_entropy(anon_pred, img_label['race'])
                result['race_ce_diff'] = anon_race_ce - orig_race_ce
                
                orig_metrics['race_losses'].append(orig_race_ce)
                anon_metrics['race_losses'].append(anon_race_ce)
            except Exception as e:
                print(f"Error in race calculation for {img_id}: {str(e)}")
            
            differences.append(result)
            total_predictions += 1
            
        except Exception as e:
            print(f"Error processing {img_id}: {str(e)}")
            continue
    
    diff_df = pd.DataFrame(differences)
    
    # 원본/익명화 통계 계산
    orig_stats = {
        'age_mae': np.mean(orig_metrics['age_errors']) if orig_metrics['age_errors'] else 0,
        'gender_acc': orig_metrics['gender_correct'] / total_predictions if total_predictions > 0 else 0,
        'gender_ce': np.mean(orig_metrics['gender_losses']) if orig_metrics['gender_losses'] else 0,
        'race_acc': orig_metrics['race_correct'] / total_predictions if total_predictions > 0 else 0,
        'race_ce': np.mean(orig_metrics['race_losses']) if orig_metrics['race_losses'] else 0
    }
    
    anon_stats = {
        'age_mae': np.mean(anon_metrics['age_errors']) if anon_metrics['age_errors'] else 0,
        'gender_acc': anon_metrics['gender_correct'] / total_predictions if total_predictions > 0 else 0,
        'gender_ce': np.mean(anon_metrics['gender_losses']) if anon_metrics['gender_losses'] else 0,
        'race_acc': anon_metrics['race_correct'] / total_predictions if total_predictions > 0 else 0,
        'race_ce': np.mean(anon_metrics['race_losses']) if anon_metrics['race_losses'] else 0
    }
    
    diff_stats = {
        'total_images': total_predictions,
        'age_error_increased': (diff_df['age_error_diff'] > 0).mean() if 'age_error_diff' in diff_df else 0,
        'age_accuracy_maintained': (diff_df['orig_age_match'] == diff_df['anon_age_match']).mean() if 'orig_age_match' in diff_df else 0,
        'gender_accuracy_maintained': (diff_df['orig_gender_match'] == diff_df['anon_gender_match']).mean() if 'orig_gender_match' in diff_df else 0,
        'race_accuracy_maintained': (diff_df['orig_race_match'] == diff_df['anon_race_match']).mean() if 'orig_race_match' in diff_df else 0,
        'avg_gender_ce_diff': diff_df['gender_ce_diff'].mean() if 'gender_ce_diff' in diff_df else 0,
        'avg_race_ce_diff': diff_df['race_ce_diff'].mean() if 'race_ce_diff' in diff_df else 0
    }
    
    # CSV 저장
    if save_csv:
        diff_df.to_csv('comparison_results.csv', index=False)
    
    return orig_stats, anon_stats, diff_stats, diff_df


# ✅ 지원하는 이미지 확장자 목록
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def is_valid_image(filename):
    """유효한 이미지 파일인지 검사"""
    return filename.lower().endswith(tuple(IMAGE_EXTENSIONS))

def process_images_to_pickle(folder_path, embedding_dict, category):
    """ 이미지 폴더를 순회하며 face_recognition을 사용하여 얼굴 벡터 추출 """
    image_count = 0
    total_images = [f for f in os.listdir(folder_path) if is_valid_image(f)]
    
    print(f"\n📂 {category} 이미지 파일 개수: {len(total_images)}")

    failed_images = []
    no_face_detected = []

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
            # Image.open(img_path).save(os.path.join(failed_image_path, filename))  # ❌ 감지 실패한 이미지 저장

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
    print(f"✅ {category} 이미지 개수: {image_count}")

    # 🔹 벡터 파일 저장 (pickle 사용)
    with open(f"{category}_embeddings.pkl", "wb") as f:
        pickle.dump(embedding_dict, f)
    print("✅ 얼굴 벡터 저장 완료!")


# ✅ 1:1 얼굴 검증 방식 (Face Verification)
def face_similarity(original_embeddings, anonymized_embeddings, threshold=0.8):
    match_count = 0
    total_faces = len(original_embeddings)
    tot_sim = 0

    for filename, orig_emb in original_embeddings.items():
        if filename in anonymized_embeddings:
            anon_emb = anonymized_embeddings[filename]
            similarity = 1 - cosine(orig_emb, anon_emb)
            tot_sim += similarity

            print(f"🔹 {filename} | 유사도: {similarity:.3f}")  # ✅ 유사도 출력
            
            if similarity > threshold:  # 임계값 이상이면 동일 인물로 판단
                match_count += 1

    reid = match_count / total_faces if total_faces > 0 else 0
    avg_sim = tot_sim / total_faces if total_faces > 0 else 0

    return reid, avg_sim


# ✅ 1:N 얼굴 식별 (Face Identification)
def face_identification_reid(original_embeddings, anonymized_embeddings, dataset_embeddings):
    match_count = 0
    total_faces = len(original_embeddings)
    total_anons = len(anonymized_embeddings)

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

    return match_count / total_faces if total_faces > 0 else 0, match_count / total_anons if total_anons > 0 else 0
