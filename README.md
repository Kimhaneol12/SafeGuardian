
# 🛡️ SafeGuardian

- 요양원에서의 낙상사고 대응 및 통합 케어 플랫폼
- '세잎클로버’처럼 소중한 일상을 지켜내는 맞춤형 안전 솔루션

---

## 🚀 주요 기능

- 다중 객체 감지
- 동일 객체 추적 및 Crop
- 다중 낙상 여부 및 유형 판정

---

## 👨‍👩‍👧‍👦 팀원 구성

| 이름 | 역할 | 이메일 |
|------|------|------|
|김한얼|영상 모델 및 통합 시스템 개발|kimhy7202@gmail.com|
|김선주|프로젝트 기획 및 문헌 조사|ssunju69@naver.com|
|김지수|센서 데이터 분석 및 모델 개발|jisso031209@gmail.com|
|백수연|연구 기획 및 데이터 분석|su1qa2ws@naver.com|
|안예진|UI/UX 디자인|yejin3308@naver.com|
|전유하|UI/UX 디자인|yhchino33@dgu.ac.kr|

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

> 🔗 [traced_model.pt 다운로드 (Google Drive)](https://drive.google.com/file/d/1AKUEylkQhEE6As2J1ZrSLhD509I8yxy1/view?usp=sharing)

1. 위 링크를 클릭하여 `traced_model.pt` 파일을 다운로드
2. 해당 파일을 프로젝트의 `src/` 디렉토리에 이동

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
