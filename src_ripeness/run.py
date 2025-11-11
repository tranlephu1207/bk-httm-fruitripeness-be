import cv2
import numpy as np
from joblib import load
from features import extract_all

models_dir = "models"
base_models = load(f"{models_dir}/feature_svms.joblib")
meta_clf = load(f"{models_dir}/meta_fusion.joblib")
with open(f"{models_dir}/classes.txt") as f:
    classes = [line.strip() for line in f if line.strip()]



img_path = "/kaggle/input/fruit-ripeness-unripe-ripe-and-rotten/fruit_ripeness_dataset/archive (1)/dataset/test/unripe orange/108.jpg"
img = cv2.imread(img_path)


img_resized = cv2.resize(img, (100, 100))
feats = extract_all(img_resized)
probs = []
for m, f in zip(base_models, feats):
    prob = m.predict_proba([f])[0]
    probs.append(prob)
meta_input = np.concatenate(probs).reshape(1, -1)
fused = meta_clf.predict_proba(meta_input)[0]
cidx = int(fused.argmax())
cscore = float(fused[cidx])
print(f"Predicted label: {classes[cidx]}, score: {cscore:.4f}")
