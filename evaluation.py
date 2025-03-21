import shutil
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from pytorch_fid import fid_score
from deepface import DeepFace
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np

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
        epsilon = 1e-15  # 로그 계산 시 0 방지를 위한 작은 값
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
            # 확률을 0~1 범위로 정규화
            normalized_prob = prob / total_prob
            normalized_prob = np.clip(normalized_prob, epsilon, 1 - epsilon)
            
            # 실제 라벨이면 1, 아니면 0
            true_value = 1.0 if race == true_race else 0.0
            
            # Cross Entropy 누적
            cross_entropy -= true_value * np.log(normalized_prob)
        
        return cross_entropy
    except Exception as e:
        print(f"Error in race cross entropy calculation: {str(e)}")
        return 0.0

def compare_agr(original_dir, anonymized_dir, label_path, save_csv=True):
    # 라벨 데이터 로드
    labels_df = pd.read_csv(label_path)
    
    # 파일 이름 형식 변환 (val/5001.jpg -> 5001_anon.jpg)
    labels_df['file'] = labels_df['file'].apply(lambda x: os.path.basename(x).split('.')[0] + '_anon.jpg')
    
    # 결과를 저장할 변수 초기화
    differences = []
    total_predictions = 0
    
    # 이미지 파일 목록 가져오기 (익명화된 이미지 기준)
    image_files = glob.glob(os.path.join(anonymized_dir, "*.jpg")) + \
                 glob.glob(os.path.join(anonymized_dir, "*.png"))
    
    # 원본과 익명화 이미지 각각의 메트릭을 저장할 변수 초기화
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
            # 파일 이름에서 ID 추출
            img_id = os.path.basename(anon_path)
            orig_path = os.path.join(original_dir, img_id.replace('_anon', ''))
            
            # 해당 이미지의 라벨 찾기
            img_label = labels_df[labels_df['file'] == img_id].iloc[0]
            
            # Original 이미지 예측
            orig_pred = DeepFace.analyze(orig_path, actions=['age', 'gender', 'race'], 
                                       align=True, detector_backend='retinaface')
            
            # Anonymized 이미지 예측
            anon_pred = DeepFace.analyze(anon_path, actions=['age', 'gender', 'race'], 
                                       align=True, detector_backend='retinaface')
            
            if not orig_pred or not anon_pred or \
               not isinstance(orig_pred, list) or not isinstance(anon_pred, list) or \
               len(orig_pred) == 0 or len(anon_pred) == 0:
                print(f"Invalid prediction format for {img_id}")
                continue
            
            orig = orig_pred[0]
            anon = anon_pred[0]
            
            # 결과 딕셔너리 초기화
            result = {
                'file_name': img_id,
                'true_age_group': img_label['age'],
                'orig_age': orig.get('age', 0),
                'anon_age': anon.get('age', 0),
                'true_gender': img_label['gender'],
                'orig_gender': convert_gender(orig.get('dominant_gender', '')),
                'anon_gender': convert_gender(anon.get('dominant_gender', '')),
                'true_race': img_label['race'],
                'orig_race': orig.get('dominant_race', '').lower(),
                'anon_race': anon.get('dominant_race', '').lower()
            }
            
            # 나이 차이 계산
            try:
                true_age = convert_age_to_number(img_label['age'])
                orig_age_error = abs(true_age - orig.get('age', 0))
                anon_age_error = abs(true_age - anon.get('age', 0))
                result['age_error_diff'] = anon_age_error - orig_age_error
                result['orig_age_match'] = orig_age_error <= 5
                result['anon_age_match'] = anon_age_error <= 5
                
                # 각각의 에러 저장
                orig_metrics['age_errors'].append(orig_age_error)
                anon_metrics['age_errors'].append(anon_age_error)
            except Exception as e:
                print(f"Error in age calculation for {img_id}: {str(e)}")
            
            # 성별 비교
            try:
                orig_gender = convert_gender(orig.get('dominant_gender', ''))
                anon_gender = convert_gender(anon.get('dominant_gender', ''))
                result['orig_gender_match'] = orig_gender.lower() == img_label['gender'].lower()
                result['anon_gender_match'] = anon_gender.lower() == img_label['gender'].lower()
                
                # 각각의 정확도 카운트
                if result['orig_gender_match']:
                    orig_metrics['gender_correct'] += 1
                if result['anon_gender_match']:
                    anon_metrics['gender_correct'] += 1
                
                # CrossEntropy 계산 및 저장
                orig_gender_ce = compute_gender_cross_entropy(orig, img_label['gender'])
                anon_gender_ce = compute_gender_cross_entropy(anon, img_label['gender'])
                result['gender_ce_diff'] = anon_gender_ce - orig_gender_ce
                
                orig_metrics['gender_losses'].append(orig_gender_ce)
                anon_metrics['gender_losses'].append(anon_gender_ce)
            except Exception as e:
                print(f"Error in gender calculation for {img_id}: {str(e)}")
            
            # 인종 비교
            try:
                mapped_true_race = race_mapping.get(img_label['race'], img_label['race'].lower())
                result['orig_race_match'] = orig.get('dominant_race', '').lower() == mapped_true_race
                result['anon_race_match'] = anon.get('dominant_race', '').lower() == mapped_true_race
                
                # 각각의 정확도 카운트
                if result['orig_race_match']:
                    orig_metrics['race_correct'] += 1
                if result['anon_race_match']:
                    anon_metrics['race_correct'] += 1
                
                # CrossEntropy 계산 및 저장
                orig_race_ce = compute_race_cross_entropy(orig, img_label['race'])
                anon_race_ce = compute_race_cross_entropy(anon, img_label['race'])
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
    
    # 결과를 DataFrame으로 변환
    diff_df = pd.DataFrame(differences)
    
    # 원본과 익명화 이미지 각각의 메트릭 계산
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
    
    # 차이에 대한 통계 계산
    diff_stats = {
        'total_images': total_predictions,
        'age_error_increased': (diff_df['age_error_diff'] > 0).mean() if 'age_error_diff' in diff_df else 0,
        'age_accuracy_maintained': (diff_df['orig_age_match'] == diff_df['anon_age_match']).mean() if 'orig_age_match' in diff_df else 0,
        'gender_accuracy_maintained': (diff_df['orig_gender_match'] == diff_df['anon_gender_match']).mean() if 'orig_gender_match' in diff_df else 0,
        'race_accuracy_maintained': (diff_df['orig_race_match'] == diff_df['anon_race_match']).mean() if 'orig_race_match' in diff_df else 0,
        'avg_gender_ce_diff': diff_df['gender_ce_diff'].mean() if 'gender_ce_diff' in diff_df else 0,
        'avg_race_ce_diff': diff_df['race_ce_diff'].mean() if 'race_ce_diff' in diff_df else 0
    }
    
    # 결과를 CSV로 저장
    if save_csv:
        diff_df.to_csv('comparison_results.csv', index=False)
    
    return orig_stats, anon_stats, diff_stats, diff_df

if __name__ == '__main__':
    # 함수 실행 예시
    orig_stats, anon_stats, diff_stats, diff_df = compare_agr('test_original', 'test_anonymized', 'fairface_label_val.csv')
    print("\nOriginal image statistics:")
    for key, value in orig_stats.items():
        print(f"{key}: {value:.4f}")
    print("\nAnonymized image statistics:")
    for key, value in anon_stats.items():
        print(f"{key}: {value:.4f}")
    print("\nDifference statistics:")
    for key, value in diff_stats.items():
        print(f"{key}: {value:.4f}")
    print("\nFirst few comparison results:")
    print(diff_df.head())