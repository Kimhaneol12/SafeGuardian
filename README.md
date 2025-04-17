
# 🛡️ SafeGuardian

- 같이 채워야 할 것

---

## 🚀 주요 기능

- 같이 채워야 할 것

---

## 👨‍👩‍👧‍👦 팀원 구성

| 이름 | 역할 | 이메일 |
|------|------|------|
|김한얼|모델 개발, 파이프라인 개발|kimhy7202@gmail.com|
|김선주|프론트엔드|      |
|김지수|데이터 분석, 모델 개발|      |
|백수연|프론트엔드|      |
|안예진|데이터 분석, 발표 자료       |      |
|전유하|데이터 분석,       |      |

---

## ⚙️ 개발 환경

| 항목 | 버전 / 정보 |
|------|-------------|
| OS | Windows 11|
| Python | 3.9 이상 |
| PyTorch | 1.7 이상 |
| CUDA | GPU 지원 가능 시 활용 |
| MediaPipe | 최신 버전 (holistic 사용) |
| 기타 | requirements.txt 참조 |

설치 명령:
```bash
pip install -r requirements.txt
```

---

## 📂 데이터 요구사항

- **fps**: 60
- **동영상 최소 길이**: 10초 이상 혹은 실시간 영상(600프레임 이상)

---

## 🧪 모델 실행 방법 (Model)

```bash
cd model/scripts/
python test.py
```

▶️ `test.py` 내부 `SOURCE` 경로만 설정  
- `'0'` : 내장 웹캠  
- `'1'`, `'2'` : 외장 USB 카메라  
- `'sample_video.mp4'` : 테스트 영상 파일  
- `'rtsp://...'`, `'http://...'` : 실시간 스트리밍 주소  

📂 **결과는 `model/eval_results/` 디렉토리에 저장**

---

## 🚀 파이프라인 실행 방법 (SafeGuardian)

```bash
cd src/
python main.py
```

▶️ `main.py` 내부 `SOURCE` 경로만 설정  
- `'0'` : 내장 웹캠  
- `'1'`, `'2'` : 외장 USB 카메라  
- `'1.mp4'` : 테스트 영상 파일
- `'rtsp://...'`, `'http://...'` : 실시간 스트리밍 주소  

📂 **Keypoint는 `outputs/keypoints/`, Crop은 `outputs/crops/` 하위에 저장**  

---

## 📁 프로젝트 파일 구조

```
SafeGuardian/
├── model/
│   ├── best_weight/
│   │   ├── fall_detection.pkl
│   │   └── fall_type.pkl
│   └── scripts/
│       ├── test.py
│       ├── scaler/
│       ├── util/
│       └── video/
│           ├── video_mediapipe.py
│           ├── video_train.py
│           └── video_evaluation.py
│           └── results/
│
├── src/
    ├── main.py
    ├── deep_sort_tracking_id.py
    ├── yolov7-tiny.pt
    └── outputs/
        ├── crops/
        └── keypoints/
```

---