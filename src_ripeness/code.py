import argparse, os, glob, random, numpy as np
import cv2
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

random.seed(42); np.random.seed(42)

def load_patches(data_root, size=100):
    X_feats=[[] for _ in range(8)]  
    y=[]
    classes=[]
    folders = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root,d))]
    if "negtrain" in folders:
        folders.remove("negtrain")
        folders.append("negtrain")
    classes = folders
    for ci, c in enumerate(classes):
        for p in glob.glob(os.path.join(data_root, c, "*")):
            img = cv2.imread(p)
            if img is None: continue
            img = cv2.resize(img, (size,size))
            feats = extract_all(img)  
            if isinstance(feats, tuple):  
                for i in range(8):
                    X_feats[i].append(feats[i])
            else:  
                pass
            y.append(ci)
    X_feats = [np.array(x) for x in X_feats]
    y = np.array(y, dtype=int)
    return X_feats, y, classes 


import os
import numpy as np
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.impute import SimpleImputer

def train_pipeline(data_root, out_dir="models", size=100, cv_folds=3, pca_dim=100):
    X_feats, y, classes = load_patches(data_root, size=size)
    os.makedirs(out_dir, exist_ok=True)

    models = []
    perfs = []
    meta_inputs = []

    # Các mô hình base
    base_models = [
        SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, class_weight="balanced", n_jobs=-1),  # HOG
        RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1),                        # LBP
        RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1),                        # Gabor
        RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1),                        # HSV Hist
        SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, class_weight="balanced", n_jobs=-1),  # Hu Moments
        RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1),                        # Haralick
        SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, class_weight="balanced", n_jobs=-1),  # Zernike
        RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1)                         # Color Moments
    ]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for i, (X, base) in enumerate(zip(X_feats, base_models)):
        print("Training Feature", i, "shape", X.shape)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA(n_components=min(pca_dim, X.shape[1]))),
            ("clf", base)
        ])

        if isinstance(base, SGDClassifier):
            clf = CalibratedClassifierCV(pipe, cv=cv_folds, method="sigmoid", n_jobs=-1)
        else:
            clf = pipe

        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)
        ypred = proba.argmax(axis=1)

        f1 = f1_score(y, ypred, average="macro")
        acc = accuracy_score(y, ypred)
        print(f"Feature {i} f1={f1:.4f} acc={acc:.4f}")
        perfs.append({"f1": f1, "acc": acc})

        meta_inputs.append(proba)

        clf.fit(X, y)
        models.append(clf)


    meta_X = np.concatenate(meta_inputs, axis=1)
    meta_y = y

    meta_clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="saga",
        n_jobs=-1
    )
    meta_clf.fit(meta_X, meta_y)

    dump(models, os.path.join(out_dir, "feature_models.joblib"))
    dump(meta_clf, os.path.join(out_dir, "meta_fusion.joblib"))
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        for c in classes:
            f.write(c + "\n")

    print("Saved base models and meta-classifier to", out_dir)
    return perfs


data_root = "/kaggle/working/data/train"
train_pipeline(data_root)
