# Fruit Ripeness Classifier

Một ứng dụng AI phân loại độ chín của trái cây từ ảnh

## Mô tả

Dự án này xây dựng một hệ thống phân loại độ chín của trái cây với khả năng nhận diện:
- **10 loại trái cây**: Apple, Banana, DragonFruit, Grape, Guava, Orange, Papaya, Pineapple, Pomegranate, Strawberry
- **3 trạng thái độ chín**: Unripe (chưa chín), Ripe (chín), Rotten (thối)

### Dataset
- **Nguồn**: [Fruit Ripeness Level Dataset](https://www.kaggle.com/datasets/dudinurdiyansah/fruit-ripeness-level-dataset)
- **Tổng số ảnh**: 24,000 ảnh (800 ảnh cho mỗi trong 30 classes)
- **Phân chia dữ liệu**: 
  - **Train**: 70% (560 ảnh/class)
  - **Validation**: 10% (80 ảnh/class) 
  - **Test**: 20% (160 ảnh/class)
- **Có thể test dữ liệu ngoài**: [Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)

## Demo

![Demo Screenshot 1](image/apple.png)
![Demo Screenshot 2](image/banana.png)

## Kiến trúc hệ thống

Dự án này so sánh 2 approaches khác nhau cho bài toán phân loại độ chín trái cây:

### 1. Deep Learning - ResNet50
- **Architecture**: ResNet50 pretrained + fine-tuning
- **Configuration**:
  - Input size: 100×100 pixels
  - Batch size: 32
  - Learning rate: 0.001
  - Optimizer: Adam
  - Epochs: 100 với Early Stopping (10 epochs)
  - Data augmentation: Resize + Normalize (ImageNet stats)

### 2. Handcrafted Features - Ensemble Learning
- **8 Base Models** cho các đặc trưng khác nhau
- **Meta-Classifier** để fusion kết quả từ các base models
- **Feature Engineering** với 8 loại đặc trưng:
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Pattern) 
  - Gabor Filters
  - HSV Color Histogram
  - Hu Moments
  - Haralick Features
  - Zernike Moments
  - Color Moments

## Cấu trúc thư mục

```
├── app.py                     # Flask web application chính
├── requirements.txt           # Dependencies Python
├── Procfile                  # Heroku deployment config
├── README.md                 # Documentation
├── src_ripeness/             # Source code của model
│   ├── features.py           # Feature extraction functions
│   ├── code.py              # Model training pipeline
│   └── run.py               # Test script
├── model_ripeness/           # Trained models
│   ├── feature_svms.joblib  # Base models cho từng feature
│   ├── meta_fusion.joblib   # Meta-classifier
│   └── classes.txt          # Danh sách 30 classes
└── model_resnet/            # Alternative CNN model (unused)
    └── cnn_on_fr.pth        # ResNet model weights
```

## Cài đặt

### Requirements
```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Chạy web application:
```bash
python app.py
```
Truy cập: `http://localhost:5000`

### 2. Authentication (JWT)

Ứng dụng dùng JWT để bảo vệ các endpoint `/predict` và `/predict_topk`.

1) Đăng ký tài khoản:
```bash
curl -X POST http://localhost:5000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"user@example.com","password":"your-password"}'
```

2) Đăng nhập lấy token:
```bash
TOKEN=$(curl -s -X POST http://localhost:5000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"user@example.com","password":"your-password"}' | jq -r '.access_token')
```

3) Gọi API với Bearer token:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -F image=@image/banana.png
```

Đặt biến môi trường `JWT_SECRET_KEY` trong production.

### 3. API Usage:
```python
import requests

# Upload ảnh và nhận prediction
token = 'REPLACE_WITH_JWT'
files = {'image': open('fruit_image.jpg', 'rb')}
headers = {'Authorization': f'Bearer {token}'}
response = requests.post('http://localhost:5000/predict', files=files, headers=headers)
result = response.json()

# Output format:
# {
#   "fruit": "apple",
#   "ripeness": "ripe", 
#   "score": 0.8945
# }
```

### 3. Train model mới:
```bash
cd src_ripeness
python code.py
```

## Thử nghiệm và So sánh

### Kết quả thử nghiệm:
1. **ResNet50**: Transfer learning với fine-tuning chỉ FC layer
2. **Handcrafted Features**: Ensemble 8 feature extractors + meta-classifier
3. **Winner**: Handcrafted approach với margin +2% accuracy

## Chi tiết kỹ thuật

### Quy trình prediction:
1. **Image Preprocessing**: Resize ảnh về 100×100 pixels
2. **Feature Extraction**: Trích xuất 8 loại đặc trưng khác nhau
3. **Base Predictions**: 8 base models predict probability
4. **Meta-Classification**: Fusion tất cả predictions
5. **Output**: Fruit type + Ripeness + Confidence score

### Base Models:
- **HOG + SVM**: Shape và texture features
- **LBP + Random Forest**: Local texture patterns  
- **Gabor + Random Forest**: Oriented texture features
- **HSV + Random Forest**: Color distribution
- **Hu Moments + SVM**: Geometric shape features
- **Haralick + Random Forest**: GLCM texture features
- **Zernike + SVM**: Shape descriptors
- **Color Moments + Random Forest**: Color statistics

### Meta-Classifier:
- **Logistic Regression** với multinomial output
- Input: Concatenation của tất cả base model probabilities
- Class balancing và regularization

## Deployment

### Heroku:
```bash
git push heroku main
```

Ứng dụng sẽ tự động deploy với cấu hình trong `Procfile`.

## Performance

```
====================== OVERALL COMPARISON ======================
                 precision    recall  f1-score   accuracy
ResNet50             0.89      0.88      0.88      0.88
HandcraftedFeatures  0.90      0.90      0.90      0.90
---------------------------------------------------------------

====================== RESNET50 CLASSIFICATION REPORT ======================
                    precision    recall  f1-score   support
RipeApple              0.85      0.89      0.87       160
RipeBanana             0.95      0.99      0.97       160
RipeDragonFruit        0.99      0.98      0.99       160
RipeGrape              0.85      0.88      0.87       160
RipeGuava              0.94      0.93      0.93       160
RipeOrange             0.89      0.96      0.92       160
RipePapaya             0.97      0.92      0.94       160
RipePineapple          0.99      0.97      0.98       160
RipePomegranate        0.98      0.87      0.88       160
RipeStrawberry         0.94      1.00      0.97       160
RottenApple            0.75      0.92      0.82       160
RottenBanana           0.92      0.96      0.94       160
RottenDragonFruit      0.99      0.95      0.97       160
RottenGrape            0.78      0.78      0.78       160
RottenGuava            0.91      0.74      0.82       160
RottenOrange           0.94      0.81      0.87       160
RottenPapaya           0.80      0.95      0.87       159
RottenPineapple        0.94      0.96      0.95       160
RottenPomegranate      0.93      0.81      0.87       160
RottenStrawberry       0.92      0.95      0.93       160
UnripeApple            0.63      0.73      0.67       160
UnripeBanana           0.99      0.85      0.91       160
UnripeDragonFruit      0.88      0.95      0.91       160
UnripeGrape            0.81      0.91      0.86       160
UnripeGuava            0.73      0.73      0.73       160
UnripeOrange           0.84      0.66      0.74       160
UnripePapaya           0.91      0.91      0.91       160
UnripePineapple        0.98      0.99      0.98       160
UnripePomegranate      0.85      0.82      0.84       160
UnripeStrawberry       0.95      0.78      0.86       160

accuracy                                   0.88      4799
macro avg              0.89      0.88      0.88      4799
weighted avg           0.89      0.88      0.88      4799
---------------------------------------------------------------

====================== HANDCRAFTED FEATURES CLASSIFICATION REPORT ======================
                    precision    recall  f1-score   support
RipeApple              0.92      0.89      0.91       160
RipeBanana             0.95      0.93      0.94       160
RipeDragonFruit        0.95      0.94      0.95       160
RipeGrape              0.85      0.83      0.89       160
RipeGuava              0.88      0.91      0.90       160
RipeOrange             0.90      0.94      0.92       160
RipePapaya             0.91      0.94      0.93       168
RipePineapple          0.99      0.94      0.97       160
RipePomegranate        0.93      0.76      0.84       160
RipeStrawberry         0.96      0.97      0.97       160
RottenApple            0.88      0.86      0.87       160
RottenBanana           0.96      0.95      0.96       160
RottenDragonFruit      1.00      0.94      0.97       160
RottenGrape            0.91      0.88      0.89       160
RottenGuava            0.84      0.92      0.88       160
RottenOrange           0.92      0.83      0.87       160
RottenPapaya           0.89      0.94      0.91       153
RottenPineapple        1.00      0.97      0.98       160
RottenPomegranate      0.91      0.91      0.91       160
RottenStrawberry       0.89      0.95      0.92       160
UnripeApple            0.79      0.86      0.82       160
UnripeBanana           0.80      0.79      0.79       160
UnripeDragonFruit      0.88      0.88      0.88       160
UnripeGrape            0.73      0.78      0.76       160
UnripeGuava            0.78      0.84      0.81       160
UnripeOrange           0.91      0.86      0.88       160
UnripePapaya           0.94      0.91      0.92       160
UnripePineapple        0.97      0.96      0.97       160
UnripePomegranate      0.90      0.88      0.89       160
UnripeStrawberry       0.89      0.92      0.90       160

accuracy                                   0.90      4799
macro avg              0.90      0.90      0.90      4799
weighted avg           0.90      0.90      0.90      4799
---------------------------------------------------------------
