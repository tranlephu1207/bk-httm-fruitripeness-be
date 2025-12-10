from flask import Flask, request, jsonify, render_template_string, send_from_directory, url_for
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flasgger import Swagger
from joblib import load, dump
from src_ripeness.features import extract_all
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import cv2
import json
import numpy as np
import os
import glob
import shutil
import time
import base64
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid4

from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'change-this-in-production')
app.config['SWAGGER'] = {'uiversion': 3}

jwt = JWTManager(app)
swagger = Swagger(app)
CORS(app, resources={r"/*": {"origins": "*"}})

models_dir = "model_ripeness"
backup_dir = "backup"
os.makedirs(backup_dir, exist_ok=True)


def get_file_info(path: str) -> Dict[str, Any]:
    """Return file existence and modified time in ISO format."""
    if not os.path.exists(path):
        return {"exists": False}
    ts = os.path.getmtime(path)
    return {
        "exists": True,
        "modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "size_bytes": os.path.getsize(path),
    }


def get_feature_models_path(model_dir: str = models_dir) -> str:
    """Get the path to the feature models file, with fallback to original."""
    new_path = os.path.join(model_dir, "feature_models_xz.joblib")
    old_path = os.path.join(model_dir, "feature_svms.joblib")
    
    # Try to load the new file first
    try:
        test_load = load(new_path)
        print(f"Using feature_models_xz.joblib")
        return new_path
    except (KeyError, ValueError, EOFError, Exception) as e:
        print(f"Warning: feature_models_xz.joblib failed to load ({e}), using feature_svms.joblib")
        return old_path


# Load base models with fallback
feature_models_path = get_feature_models_path()
base_models = load(feature_models_path)

meta_clf = load(f"{models_dir}/meta_fusion.joblib")
with open(f"{models_dir}/classes.txt") as f:
    classes = [line.strip() for line in f if line.strip()]

# ==== ResNet model (for comparison) ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    state = torch.load("model_resnet/cnn_on_fr.pth", map_location=device)
    num_classes = state["fc.weight"].shape[0]
    resnet_model = models.resnet50()
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    resnet_model.load_state_dict(state)
    resnet_model.to(device)
    resnet_model.eval()
    print(f"ResNet model loaded successfully on {device}")
except Exception as e:
    print(f"Warning: Failed to load ResNet model: {e}")
    resnet_model = None

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))
UPLOAD_DIR = Path(os.environ.get('UPLOAD_DIR', DATA_DIR / 'uploads'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FRUITS_FILE = DATA_DIR / 'fruits.json'
SCANS_FILE = DATA_DIR / 'scans.json'
ORDERS_FILE = DATA_DIR / 'orders.json'
USERS_FILE = DATA_DIR / 'users.json'
BATCH_HISTORY_FILE = DATA_DIR / 'batch_history.json'

DEFAULT_FRUITS: List[Dict[str, Any]] = [
    {
        "id": "1",
        "name": "Red Apple",
        "description": "Crisp and sweet red apples, perfect for snacking.",
        "price": 1.99,
        "image": "https://images.unsplash.com/photo-1623815242959-fb20354f9b8d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVzaCUyMHJlZCUyMGFwcGxlJTIwZnJ1aXR8ZW58MXx8fHwxNzYyMjcyOTA4fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        "stock": 500,
        "sold": 342,
        "remaining": 158,
        "updated_at": "2024-11-01T00:00:00Z"
    },
    {
        "id": "2",
        "name": "Banana",
        "description": "Fresh yellow bananas, rich in potassium.",
        "price": 0.89,
        "image": "https://images.unsplash.com/photo-1573828235229-fb27fdc8da91?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVzaCUyMGJhbmFuYSUyMGZydWl0fGVufDF8fHx8MTc2MjI3MjkwOXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        "stock": 400,
        "sold": 289,
        "remaining": 111,
        "updated_at": "2024-11-01T00:00:00Z"
    },
    {
        "id": "3",
        "name": "Orange",
        "description": "Juicy oranges packed with vitamin C.",
        "price": 1.49,
        "image": "https://images.unsplash.com/photo-1668617596950-a18f6a720c58?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVzaCUyMG9yYW5nZSUyMGNpdHJ1c3xlbnwxfHx8fDE3NjIyNzI5MDh8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        "stock": 350,
        "sold": 198,
        "remaining": 152,
        "updated_at": "2024-11-01T00:00:00Z"
    },
    {
        "id": "4",
        "name": "Grapes",
        "description": "Sweet purple grapes, great for snacking or making juice.",
        "price": 3.99,
        "image": "https://images.unsplash.com/photo-1596363505729-4190a9506133?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmcmVzaCUyMGdyYXBlcyUyMGZydWl0fGVufDF8fHx8MTc2MjI3MjkwOXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        "stock": 300,
        "sold": 245,
        "remaining": 55,
        "updated_at": "2024-11-01T00:00:00Z"
    }
]


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def load_json(path: Path, default: Any) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(default, indent=2))
        return _deep_copy(default)
    with path.open('r') as fh:
        return json.load(fh)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_fruits_file() -> List[Dict[str, Any]]:
    return load_json(FRUITS_FILE, DEFAULT_FRUITS)


def save_fruits_file(data: List[Dict[str, Any]]) -> None:
    save_json(FRUITS_FILE, data)


def load_scans_file() -> List[Dict[str, Any]]:
    return load_json(SCANS_FILE, [])


def save_scans_file(data: List[Dict[str, Any]]) -> None:
    save_json(SCANS_FILE, data)


def load_orders_file() -> List[Dict[str, Any]]:
    return load_json(ORDERS_FILE, [])


def save_orders_file(data: List[Dict[str, Any]]) -> None:
    save_json(ORDERS_FILE, data)


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)


def load_batch_history() -> List[Dict[str, Any]]:
    """Load batch merge/reject history."""
    return load_json(BATCH_HISTORY_FILE, [])


def save_batch_history(data: List[Dict[str, Any]]) -> None:
    """Persist batch merge/reject history."""
    save_json(BATCH_HISTORY_FILE, data)


PENDING_BATCHES_FILE = DATA_DIR / 'pending_batches.json'


def load_pending_batches() -> Dict[str, Any]:
    """Load pending batches awaiting admin confirmation."""
    return load_json(PENDING_BATCHES_FILE, {})


def save_pending_batches(data: Dict[str, Any]) -> None:
    """Persist pending batches."""
    save_json(PENDING_BATCHES_FILE, data)


def init_admin_account():
    """Initialize admin account on first run if it doesn't exist"""
    users = load_users()
    admin_email = 'admin@fruit.com'
    if admin_email not in users:
        admin_password = 'admin'
        password_hash = generate_password_hash(admin_password)
        users[admin_email] = {'password_hash': password_hash}
        save_users(users)
        print(f"Admin account created: email='{admin_email}', password='{admin_password}'")
    else:
        print(f"Admin account already exists: email='{admin_email}'")


def current_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec='seconds') + 'Z'


def parse_label(label: str) -> Tuple[str, str]:
    label = label.strip().lower()
    ripeness = "unknown"
    fruit = label
    for prefix in ["ripe", "rotten", "unripe"]:
        if label.startswith(prefix):
            ripeness = prefix
            fruit = label[len(prefix):]
            break
    fruit = fruit.replace(" ", "")
    return ripeness, fruit


def convert_to_class_label(fruit_name: str, ripeness: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
    """
    Map a (fruit, ripeness) pair to a class index/name using the loaded classes.txt.
    Falls back to fruit-only match if ripeness is missing.
    """
    if not fruit_name:
        return None, None
    fruit_clean = fruit_name.replace(" ", "").lower()
    ripeness_clean = (ripeness or "").replace(" ", "").lower()
    if ripeness_clean and ripeness_clean not in {"ripe", "unripe", "rotten"}:
        ripeness_clean = ""

    candidates = []
    if ripeness_clean:
        candidates.append(f"{ripeness_clean}{fruit_clean}")
    candidates.append(fruit_clean)

    for candidate in candidates:
        for idx, cls in enumerate(classes):
            if cls.lower() == candidate:
                return idx, cls

    # Fallback: match by fruit only even if class has ripeness prefix
    for idx, cls in enumerate(classes):
        _, cls_fruit = parse_label(cls)
        if cls_fruit == fruit_clean:
            return idx, cls

    return None, None


def predict_with_resnet(img_array: np.ndarray) -> Tuple[int, float]:
    """
    Predict using the ResNet model on a cv2 image array.
    Returns (class_index, confidence_score) or (None, None) if unavailable.
    """
    if resnet_model is None:
        return None, None

    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_tensor = resnet_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = resnet_model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, class_idx = torch.max(probs, 0)
        return int(class_idx), float(confidence)
    except Exception as e:
        print(f"Error in ResNet prediction: {e}")
        return None, None


def load_patches(data_root: str, size: int = 100):
    X_feats = [[] for _ in range(8)]
    y = []
    classes_local = []
    folders = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d))]
    if "negtrain" in folders:
        folders.remove("negtrain")
        folders.append("negtrain")
    classes_local = folders
    for ci, c in enumerate(classes_local):
        for p in glob.glob(os.path.join(data_root, c, "*")):
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.resize(img, (size, size))
            feats = extract_all(img)
            if isinstance(feats, tuple):
                for i in range(8):
                    X_feats[i].append(feats[i])
            y.append(ci)
    X_feats = [np.array(x) for x in X_feats]
    y = np.array(y, dtype=int)
    return X_feats, y, classes_local


def evaluate(test_root: str, model_dir: str = models_dir, size: int = 100):
    feature_models_path = get_feature_models_path(model_dir)
    feature_models = load(feature_models_path)
    meta = load(os.path.join(model_dir, "meta_fusion.joblib"))
    with open(os.path.join(model_dir, "classes.txt")) as f:
        eval_classes = [line.strip() for line in f if line.strip()]

    X_feats, y_true, _ = load_patches(test_root, size=size)

    meta_inputs = []
    for X, model in zip(X_feats, feature_models):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:
            ypred = model.predict(X)
            proba = np.eye(len(eval_classes))[ypred]
        meta_inputs.append(proba)
    meta_X = np.concatenate(meta_inputs, axis=1)

    imputer = SimpleImputer(strategy="constant", fill_value=0)
    meta_X = imputer.fit_transform(meta_X)

    y_pred = meta.predict(meta_X)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


def update_meta_model(
    meta_model_path: str,
    feature_models_path: str,
    classes_path: str,
    new_images: List[np.ndarray],
    new_labels: List[int],
    size: int = 100,
    alpha: float = 0.1,
):
    """
    Online update for meta-classifier (SGDClassifier) with safe blending, adapted
    from IS_Assignment_251.
    """
    feature_models = load(feature_models_path)
    meta = load(meta_model_path)

    with open(classes_path) as f:
        cls = [line.strip() for line in f]
    n_classes = len(cls)
    classes_arr = np.arange(0, n_classes)

    meta_inputs = []
    for img in new_images:
        if isinstance(img, str):
            img = cv2.imread(img)
        if img is None:
            raise ValueError("Invalid image for update")
        img = cv2.resize(img, (size, size))
        feats = extract_all(img)
        feature_probas = []
        for i, model in enumerate(feature_models):
            X_feat = np.array(feats[i]).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_feat)
            else:
                pred = model.predict(X_feat)
                proba = np.eye(n_classes)[pred]
            feature_probas.append(proba)
        meta_input = np.concatenate(feature_probas, axis=1)
        meta_inputs.append(meta_input)

    meta_X = np.vstack(meta_inputs)
    meta_y = np.array(new_labels, dtype=int).ravel()

    if hasattr(meta, "coef_") and hasattr(meta, "intercept_"):
        old_coef = meta.coef_.copy()
        old_intercept = meta.intercept_.copy()
    else:
        old_coef = old_intercept = None

    unique_labels = np.unique(meta_y)
    try:
        if len(unique_labels) == 1:
            tmp_clf = SGDClassifier(
                loss=meta.loss,
                max_iter=meta.max_iter,
                learning_rate=meta.learning_rate,
                eta0=meta.eta0,
                n_jobs=-1,
                random_state=meta.random_state,
                class_weight=None,
            )
            tmp_clf.classes_ = meta.classes_
            tmp_clf.coef_ = meta.coef_.copy()
            tmp_clf.intercept_ = meta.intercept_.copy()
            tmp_clf.partial_fit(meta_X, meta_y, classes=classes_arr)
            meta = tmp_clf
        else:
            meta.partial_fit(meta_X, meta_y)
    except ValueError:
        tmp_clf = SGDClassifier(
            loss=meta.loss,
            max_iter=meta.max_iter,
            learning_rate=meta.learning_rate,
            eta0=meta.eta0,
            n_jobs=-1,
            random_state=meta.random_state,
            class_weight=None,
        )
        tmp_clf.classes_ = meta.classes_
        tmp_clf.coef_ = meta.coef_.copy()
        tmp_clf.intercept_ = meta.intercept_.copy()
        tmp_clf.partial_fit(meta_X, meta_y, classes=classes_arr)
        meta = tmp_clf

    if old_coef is not None and hasattr(meta, "coef_"):
        meta.coef_ = (1 - alpha) * old_coef + alpha * meta.coef_
        meta.intercept_ = (1 - alpha) * old_intercept + alpha * meta.intercept_

    dump(meta, meta_model_path)


def run_inference(image_bytes: bytes, topk: int = 5) -> List[Dict[str, Any]]:
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot read image")
    img_resized = cv2.resize(img, (100, 100))
    feats = extract_all(img_resized)
    probs = []
    for model, feat in zip(base_models, feats):
        prob = model.predict_proba([feat])[0]
        probs.append(prob)
    meta_input = np.concatenate(probs).reshape(1, -1)
    fused = meta_clf.predict_proba(meta_input)[0]
    indices = np.argsort(fused)[::-1][:topk]
    results = []
    for idx in indices:
        label = classes[int(idx)]
        ripeness, fruit = parse_label(label)
        results.append({
            'label': label,
            'fruit': fruit,
            'ripeness': ripeness,
            'score': round(float(fused[int(idx)]), 4)
        })
    return results


def record_scan(image_bytes: bytes, original_filename: str, topk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scans = load_scans_file()
    scan_id = uuid4().hex
    filename = secure_filename(original_filename or f"{scan_id}.jpg")
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = '.jpg'
    stored_name = f"{scan_id}{ext}"
    upload_path = UPLOAD_DIR / stored_name
    upload_path.write_bytes(image_bytes)

    entry = {
        'id': scan_id,
        'timestamp': current_timestamp(),
        'image_url': f"/uploads/{stored_name}",
        'topk': topk_results,
        'user_agreed': None,
        'user_selected_fruit': None,
        'admin_corrected_fruit': None,
        'saved': False
    }
    scans.insert(0, entry)
    save_scans_file(scans)
    return entry


HTML_FORM = """
<!doctype html>
<title>Fruit Ripeness Classifier</title>
<h2>Upload an image to classify ripeness</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=image>
  <input type=submit value=Upload>
</form>
<a href="/update"><button>Update Model</button></a>
<a href="/compare"><button>Compare Models</button></a>
{% if result %}
  <h3>Result:</h3>
  <b>Fruit:</b> {{ result['fruit'] }}<br>
  <b>Ripeness:</b> {{ result['ripeness'] }}<br>
  <b>Score:</b> {{ result['score'] }}
{% endif %}
"""


HTML_UPDATE_FORM = """
<!doctype html>
<title>Update Fruit Ripeness Model</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; }
  .container { max-width: 600px; }
  h2 { color: #333; }
  form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
  input[type="file"] { margin: 10px 0; }
  input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
  input[type="submit"]:hover { background-color: #45a049; }
  .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
  .accepted { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
  .rejected { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
  .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
  .metrics { margin: 10px 0; font-family: monospace; }
</style>
<div class="container">
  <h2>Update Fruit Ripeness Model</h2>
  <p>Upload an image to update and train the model with new data.</p>
  <form method=post enctype=multipart/form-data>
    <label for="image">Select image:</label><br>
    <input type=file name=image required>
    <input type=submit value="Update Model">
  </form>
  
  {% if result %}
    {% if result.get('error') %}
      <div class="result error">
        <h3>Error:</h3>
        <p>{{ result['error'] }}</p>
      </div>
    {% elif result.get('result') == 'accepted' %}
      <div class="result accepted">
        <h3>Model Update Accepted</h3>
        <p>The model has been successfully updated with improved performance.</p>
        <div class="metrics">
          <b>Old Accuracy:</b> {{ "%.4f" % result['old_acc'] }}<br>
          <b>New Accuracy:</b> {{ "%.4f" % result['new_acc'] }}<br>
        </div>
      </div>
    {% elif result.get('result') == 'rejected' %}
      <div class="result rejected">
        <h3>Model Update Rejected</h3>
        <p>The update was rejected. Original model restored.</p>
        <div class="metrics">
          <b>Old Accuracy:</b> {{ "%.4f" % result['old_acc'] }}<br>
          <b>New Accuracy:</b> {{ "%.4f" % result['new_acc'] }}<br>
        </div>
      </div>
    {% endif %}
  {% endif %}
</div>
"""


HTML_FORM_COMPARE = """
<!doctype html>
<title>Compare Models</title>
<h2>Upload an image to compare models</h2>

<form method="post" enctype="multipart/form-data">
  <input type="file" name="image">
  <input type="submit" value="Upload">
</form>

{% if result %}
  <img src="{{ result['image'] }}" alt="Uploaded analysis result" style="max-width: 300px; display:block; margin-top: 16px;">
  <h3>Meta Fusion Model</h3>
  <b>Name:</b> {{ result['current_model']['name'] }}<br>
  <b>Fruit:</b> {{ result['current_model']['fruit'] }}<br>
  <b>Ripeness:</b> {{ result['current_model']['ripeness'] }}<br>
  <b>Score:</b> {{ result['current_model']['score'] }}<br>
  <b>Accuracy (approx):</b> {{ result['current_model']['accuracy'] }}%<br>
  <b>Time:</b> {{ result['current_model']['time'] }} s<br>

  <hr>

  <h3>ResNet Model</h3>
  <b>Name:</b> {{ result['resnet_model']['name'] }}<br>
  <b>Fruit:</b> {{ result['resnet_model']['fruit'] }}<br>
  <b>Ripeness:</b> {{ result['resnet_model']['ripeness'] }}<br>
  <b>Score:</b> {{ result['resnet_model']['score'] }}<br>
  <b>Accuracy (approx):</b> {{ result['resnet_model']['accuracy'] }}%<br>
  <b>Time:</b> {{ result['resnet_model']['time'] }} s<br>
  <hr>
{% endif %}
"""


@app.route('/uploads/<path:filename>')
def serve_upload(filename: str):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'error': 'email and password are required'}), 400

    users = load_users()
    if email in users:
        return jsonify({'error': 'email already registered'}), 409

    password_hash = generate_password_hash(password)
    users[email] = {'password_hash': password_hash}
    save_users(users)
    return jsonify({'message': 'registered successfully'}), 201


@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'error': 'email and password are required'}), 400

    users = load_users()
    user = users.get(email)
    if not user or not check_password_hash(user.get('password_hash', ''), password):
        return jsonify({'error': 'invalid credentials'}), 401

    token = create_access_token(identity=email)
    return jsonify({'access_token': token}), 200


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None:
            result = {'error': 'No image uploaded'}
        else:
            raw = file.read()
            try:
                topk = run_inference(raw, topk=5)
            except ValueError as exc:
                result = {'error': str(exc)}
            else:
                record_scan(raw, file.filename, topk)
                best = topk[0] if topk else None
                if not best:
                    result = {'error': 'No prediction generated'}
                else:
                    result = {
                        'fruit': best['fruit'],
                        'ripeness': best['ripeness'],
                        'score': best['score']
                    }
    return render_template_string(HTML_FORM, result=result)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'No image uploaded'}), 400
    raw = file.read()
    try:
        topk = run_inference(raw, topk=1)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    record_scan(raw, file.filename, topk)
    best = topk[0]
    response = {
        'fruit': best['fruit'],
        'ripeness': best['ripeness'],
        'score': best['score']
    }
    return jsonify(response)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """
    Compare predictions between the current Meta Fusion model and ResNet
    using the same image, similar to IS_Assignment_251.
    """
    result = None
    if request.method == 'POST':
        if resnet_model is None:
            return jsonify({'error': 'ResNet model not available'}), 500

        file = request.files.get('image')
        if file is None:
            return jsonify({'error': 'No image uploaded'}), 400

        raw = file.read()
        file_bytes = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Cannot read image'}), 400

        # Save uploaded file to UPLOAD_DIR for visualization
        scan_id = uuid4().hex
        filename = secure_filename(file.filename or f"{scan_id}.jpg")
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".jpg"
        stored_name = f"{scan_id}{ext}"
        upload_path = UPLOAD_DIR / stored_name
        upload_path.write_bytes(raw)
        image_url = url_for('serve_upload', filename=stored_name)

        # Current Meta Fusion model
        start_time_current = time.time()
        img_resized = cv2.resize(img, (100, 100))
        feats = extract_all(img_resized)
        probs = []
        for m, f in zip(base_models, feats):
            prob = m.predict_proba([f])[0]
            probs.append(prob)
        meta_input = np.concatenate(probs).reshape(1, -1)
        fused = meta_clf.predict_proba(meta_input)[0]
        current_cidx = int(fused.argmax())
        current_score = float(fused[current_cidx])
        current_label = classes[current_cidx]
        current_ripeness, current_fruit = parse_label(current_label)
        current_time = time.time() - start_time_current

        # ResNet model
        start_time_resnet = time.time()
        resnet_cidx, resnet_score = predict_with_resnet(img)
        resnet_time = time.time() - start_time_resnet

        if resnet_cidx is not None and resnet_cidx < len(classes):
            resnet_label = classes[resnet_cidx]
            resnet_ripeness, resnet_fruit = parse_label(resnet_label)
        else:
            resnet_label = "unknown"
            resnet_ripeness = "unknown"
            resnet_fruit = "unknown"
            resnet_score = 0.0

        current_accuracy = round(current_score * 100, 2)
        resnet_accuracy = round(resnet_score * 100, 2) if resnet_score is not None else 0.0

        result = {
            'image': image_url,
            'current_model': {
                'name': 'Meta Fusion',
                'fruit': current_fruit,
                'ripeness': current_ripeness,
                'score': round(current_score, 4),
                'accuracy': current_accuracy,
                'time': round(current_time, 2),
            },
            'resnet_model': {
                'name': 'ResNet',
                'fruit': resnet_fruit,
                'ripeness': resnet_ripeness,
                'score': round(resnet_score, 4),
                'accuracy': resnet_accuracy,
                'time': round(resnet_time, 2),
            },
        }

    return render_template_string(HTML_FORM_COMPARE, result=result)


@app.route('/update', methods=['GET', 'POST'])
def update():
    """
    Online learning endpoint adapted from IS_Assignment_251.
    Uses images in image/temp as pre-update and evaluates on image/ as test root.
    """
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            result = {'error': 'No image uploaded'}
        else:
            file = request.files['image']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = [cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)]
            if img[0] is None:
                result = {'error': 'Cannot read image'}
            else:
                label = [0]
                meta_model_path = os.path.join(models_dir, 'meta_fusion.joblib')
                feature_models_path = get_feature_models_path(models_dir)
                classes_path = os.path.join(models_dir, 'classes.txt')
                test_root = os.path.join(os.getcwd(), 'image')
                fruits_dir = os.path.join(test_root, 'temp')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, 'meta_fusion.joblib')

                shutil.copy(meta_model_path, backup_path)
                try:
                    old_acc, old_f1 = evaluate(test_root, model_dir=models_dir, size=100)
                except Exception as e:
                    result = {'error': f'Failed to evaluate old model: {str(e)}'}
                    return render_template_string(HTML_UPDATE_FORM, result=result)

                # Pre-training with existing temp images
                if os.path.isdir(fruits_dir):
                    for filename in os.listdir(fruits_dir):
                        image_path = os.path.join(fruits_dir, filename)
                        image_f = [cv2.imread(image_path, cv2.IMREAD_COLOR)]
                        label_f = [0]
                        if image_f[0] is None:
                            continue
                        update_meta_model(
                            meta_model_path,
                            feature_models_path,
                            classes_path,
                            image_f,
                            label_f,
                            size=100,
                            alpha=0.1,
                        )
                        new_acc_f, new_f1_f = evaluate(test_root, model_dir=models_dir, size=100)
                        if new_acc_f <= old_acc:
                            shutil.copy(backup_path, meta_model_path)

                # Main update with user image
                try:
                    update_meta_model(
                        meta_model_path,
                        feature_models_path,
                        classes_path,
                        img,
                        label,
                        size=100,
                        alpha=0.1,
                    )
                except Exception as e:
                    result = {'error': f'Update failed: {str(e)}'}
                    return render_template_string(HTML_UPDATE_FORM, result=result)

                try:
                    new_acc, new_f1 = evaluate(test_root, model_dir=models_dir, size=100)
                except Exception as e:
                    result = {'error': f'Failed to evaluate new model: {str(e)}'}
                    return render_template_string(HTML_UPDATE_FORM, result=result)

                if new_acc > old_acc:
                    result = {
                        'result': 'accepted',
                        'old_acc': old_acc,
                        'new_acc': new_acc,
                    }
                else:
                    shutil.copy(backup_path, meta_model_path)
                    os.makedirs(fruits_dir, exist_ok=True)
                    fname = f'img_{int(time.time())}.jpg'
                    cv2.imwrite(os.path.join(fruits_dir, fname), img[0])
                    result = {
                        'result': 'rejected',
                        'old_acc': old_acc,
                        'new_acc': new_acc,
                    }

    return render_template_string(HTML_UPDATE_FORM, result=result)

@app.route('/predict_topk', methods=['POST'])
def predict_topk():
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'No image uploaded'}), 400
    raw = file.read()
    try:
        results = run_inference(raw, topk=5)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    scan_entry = record_scan(raw, file.filename, results)
    return jsonify({
        'scan_id': scan_entry['id'],
        'image_url': scan_entry['image_url'],
        'topk': results
    })


@app.route('/fruits', methods=['GET'])
def get_fruits():
    fruits = load_fruits_file()
    stats = {
        'total_stock': sum(f.get('stock', 0) for f in fruits),
        'total_sold': sum(f.get('sold', 0) for f in fruits),
        'total_remaining': sum(f.get('remaining', 0) for f in fruits),
        'updated_at': max((f.get('updated_at') or '') for f in fruits) if fruits else None
    }
    return jsonify({'fruits': fruits, 'stats': stats})


@app.route('/fruits', methods=['POST'])
@jwt_required()
def create_fruit():
    data = request.get_json(silent=True) or {}
    required_fields = ['name', 'description', 'price', 'image']
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        return jsonify({'error': f"Missing fields: {', '.join(missing)}"}), 400

    fruits = load_fruits_file()
    new_fruit = {
        'id': uuid4().hex,
        'name': data['name'],
        'description': data['description'],
        'price': float(data.get('price', 0)),
        'image': data['image'],
        'stock': int(data.get('stock', 0)),
        'sold': int(data.get('sold', 0)),
        'remaining': int(data.get('remaining', data.get('stock', 0))),
        'updated_at': current_timestamp()
    }
    fruits.append(new_fruit)
    save_fruits_file(fruits)
    return jsonify(new_fruit), 201


@app.route('/fruits/<fruit_id>', methods=['PATCH'])
@jwt_required()
def update_fruit(fruit_id: str):
    data = request.get_json(silent=True) or {}
    fruits = load_fruits_file()
    for fruit in fruits:
        if fruit['id'] == fruit_id:
            for field in ['name', 'description', 'image']:
                if field in data:
                    fruit[field] = data[field]
            if 'price' in data:
                fruit['price'] = float(data['price'])
            if 'stock' in data:
                fruit['stock'] = max(0, int(data['stock']))
            if 'sold' in data:
                fruit['sold'] = max(0, int(data['sold']))
            if 'remaining' in data:
                fruit['remaining'] = max(0, int(data['remaining']))
            if 'stock_delta' in data:
                delta = int(data['stock_delta'])
                fruit['stock'] = max(0, fruit.get('stock', 0) + delta)
                fruit['remaining'] = max(0, fruit.get('remaining', 0) + delta)
            fruit['updated_at'] = current_timestamp()
            save_fruits_file(fruits)
            return jsonify(fruit)
    return jsonify({'error': 'fruit not found'}), 404


@app.route('/orders', methods=['GET'])
@jwt_required()
def get_orders():
    orders = load_orders_file()
    email = get_jwt_identity()
    user_orders = [order for order in orders if order.get('user_email') == email]
    return jsonify({'orders': user_orders})


@app.route('/orders', methods=['POST'])
@jwt_required()
def create_order():
    identity = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    items = data.get('items') or []
    if not items:
        return jsonify({'error': 'Order must include at least one item'}), 400

    fruits = load_fruits_file()
    fruit_by_id = {fruit['id']: fruit for fruit in fruits}
    order_items = []
    subtotal = 0.0

    for item in items:
        fruit_id = item.get('fruit_id')
        quantity = int(item.get('quantity', 0))
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be greater than zero'}), 400
        fruit = fruit_by_id.get(fruit_id)
        if not fruit:
            return jsonify({'error': f'Fruit {fruit_id} not found'}), 404
        if fruit.get('remaining', 0) < quantity:
            return jsonify({'error': f'Not enough inventory for {fruit["name"]}'}), 400

        line_total = round(float(fruit['price']) * quantity, 2)
        subtotal += line_total
        fruit['sold'] = fruit.get('sold', 0) + quantity
        fruit['remaining'] = max(0, fruit.get('remaining', 0) - quantity)
        order_items.append({
            'fruit_id': fruit_id,
            'name': fruit['name'],
            'quantity': quantity,
            'unit_price': fruit['price'],
            'line_total': line_total
        })

    tax = round(subtotal * 0.10, 2)
    total = round(subtotal + tax, 2)

    orders = load_orders_file()
    order = {
        'id': uuid4().hex,
        'user_email': identity,
        'items': order_items,
        'subtotal': round(subtotal, 2),
        'tax': tax,
        'total': total,
        'created_at': current_timestamp()
    }
    orders.append(order)
    save_orders_file(orders)
    save_fruits_file(fruits)
    return jsonify(order), 201


@app.route('/scans', methods=['GET'])
def get_scans():
    limit = int(request.args.get('limit', 50))
    scans = load_scans_file()[:limit]
    return jsonify({'scans': scans})


@app.route('/scans/<scan_id>', methods=['PATCH'])
@jwt_required()
def update_scan(scan_id: str):
    data = request.get_json(silent=True) or {}
    scans = load_scans_file()
    for scan in scans:
        if scan['id'] == scan_id:
            if 'user_agreed' in data:
                scan['user_agreed'] = data['user_agreed']
            if 'user_selected_fruit' in data:
                scan['user_selected_fruit'] = data['user_selected_fruit']
            if 'admin_corrected_fruit' in data:
                scan['admin_corrected_fruit'] = data['admin_corrected_fruit']
            if 'saved' in data:
                scan['saved'] = data['saved']
            save_scans_file(scans)
            return jsonify(scan)
    return jsonify({'error': 'scan not found'}), 404


@app.route('/scans/<scan_id>/feedback', methods=['POST'])
@jwt_required()
def record_feedback(scan_id: str):
    data = request.get_json(silent=True) or {}
    if 'user_agreed' not in data:
        return jsonify({'error': 'user_agreed is required'}), 400
    scans = load_scans_file()
    for scan in scans:
        if scan['id'] == scan_id:
            scan['user_agreed'] = bool(data['user_agreed'])
            scan['user_selected_fruit'] = data.get('user_selected_fruit')
            scan['saved'] = True
            save_scans_file(scans)
            return jsonify(scan)
    return jsonify({'error': 'scan not found'}), 404


@app.route('/analytics/summary', methods=['GET'])
def analytics_summary():
    scans = load_scans_file()
    total = len(scans)
    user_agreed = sum(1 for scan in scans if scan.get('user_agreed') is True)
    user_disagreed = sum(1 for scan in scans if scan.get('user_agreed') is False)
    admin_corrected = sum(1 for scan in scans if scan.get('admin_corrected_fruit'))

    daily_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'total': 0, 'agreed': 0, 'disagreed': 0})
    fruit_breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: {'count': 0, 'agreed': 0, 'disagreed': 0})
    samples_by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for scan in scans:
        timestamp = scan.get('timestamp') or ''
        date_key = timestamp.split('T')[0] if 'T' in timestamp else timestamp[:10]
        entry = daily_stats[date_key]
        entry['total'] += 1
        fruit_name = (scan.get('topk') or [{}])[0].get('fruit', 'unknown')
        fruit_entry = fruit_breakdown[fruit_name]
        fruit_entry['count'] += 1
        agreed = scan.get('user_agreed')
        if agreed is True:
            entry['agreed'] += 1
            fruit_entry['agreed'] += 1
        elif agreed is False:
            entry['disagreed'] += 1
            fruit_entry['disagreed'] += 1

        # collect samples for UI
        top_pred = (scan.get('topk') or [{}])[0]
        predicted_label = top_pred.get('label')
        predicted_fruit = top_pred.get('fruit')
        predicted_ripeness = top_pred.get('ripeness')

        user_label = None
        user_fruit = None
        user_ripeness = None
        if agreed is True and predicted_label:
            user_label = predicted_label
            user_fruit = predicted_fruit
            user_ripeness = predicted_ripeness
        elif scan.get('user_selected_fruit'):
            user_label = scan.get('user_selected_fruit')
            user_fruit = scan.get('user_selected_fruit')
            user_ripeness = predicted_ripeness

        admin_label = scan.get('admin_corrected_fruit') or None
        admin_fruit = admin_label
        admin_ripeness = predicted_ripeness if admin_label else None

        samples_by_date[date_key].append({
            'id': scan.get('id'),
            'image_url': scan.get('image_url'),
            'predicted_label': predicted_label,
            'predicted_fruit': predicted_fruit,
            'predicted_ripeness': predicted_ripeness,
            'user_label': user_label,
            'user_fruit': user_fruit,
            'user_ripeness': user_ripeness,
            'admin_label': admin_label,
            'admin_fruit': admin_fruit,
            'admin_ripeness': admin_ripeness,
            'user_agreed': agreed,
            'timestamp': scan.get('timestamp'),
        })

    overall_accuracy = (user_agreed / total) if total else 0

    daily = [
        {
            'date': date,
            'total': stats['total'],
            'correct': stats['agreed'],
            'incorrect': stats['disagreed'],
            'accuracy': round((stats['agreed'] / stats['total']) * 100, 2) if stats['total'] else 0,
            'samples': samples_by_date.get(date, [])[:50],
        }
        for date, stats in sorted(daily_stats.items(), reverse=True)
    ]

    fruits = [
        {
            'fruit': fruit,
            'total': stats['count'],
            'correct': stats['agreed'],
            'incorrect': stats['disagreed'],
            'accuracy': round((stats['agreed'] / stats['count']) * 100, 2) if stats['count'] else 0
        }
        for fruit, stats in fruit_breakdown.items()
    ]

    return jsonify({
        'counts': {
            'total_scans': total,
            'user_agreed': user_agreed,
            'user_disagreed': user_disagreed,
            'admin_corrected': admin_corrected,
            'overall_accuracy': round(overall_accuracy * 100, 2) if total else 0
        },
        'daily': daily,
        'fruits': fruits
    })


@app.route('/update_batch', methods=['POST'])
def update_batch():
    """
    Receives a list of labeled images and trains the model for preview.
    By default, returns comparison metrics for admin review without auto-deciding.
    Admin must call /confirm_batch to finalize merge/reject.
    
    Expected JSON:
    {
      "images": [...],
      "batch_id": "...",
      "force": false,        # Override previously processed batch
      "auto_decide": false   # If true, auto-merge/reject based on accuracy (legacy behavior)
    }
    """
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()
    images_data = data.get('images') or data.get('samples')
    batch_id = data.get('batch_id') or data.get('batchId')
    force_merge = bool(data.get('force'))
    auto_decide = bool(data.get('auto_decide'))  # Legacy mode: auto merge/reject

    if not batch_id or not isinstance(batch_id, str):
        return jsonify({'error': '"batch_id" is required'}), 400

    if not images_data or not isinstance(images_data, list):
        return jsonify({'error': '"images" must be a non-empty list'}), 400

    # Check batch history
    batch_history = load_batch_history()
    previous_entries = [entry for entry in batch_history if entry.get('batch_id') == batch_id]
    
    # Check pending batches
    pending_batches = load_pending_batches()
    
    # If already finalized and not forcing, return the existing result
    if previous_entries and not force_merge:
        last_entry = previous_entries[-1]
        return jsonify({
            'result': last_entry.get('status'),
            'batch_id': batch_id,
            'status': last_entry.get('status'),
            'old_acc': last_entry.get('metrics_before', {}).get('acc'),
            'new_acc': last_entry.get('metrics_after', {}).get('acc'),
            'old_f1': last_entry.get('metrics_before', {}).get('f1'),
            'new_f1': last_entry.get('metrics_after', {}).get('f1'),
            'merged_at': last_entry.get('merged_at'),
            'rejected_at': last_entry.get('rejected_at'),
            'train_time_seconds': last_entry.get('train_time_seconds'),
            'note': 'Batch already finalized. Use force=true to reprocess.'
        }), 200
    
    # If already pending review, return the pending state
    if batch_id in pending_batches and not force_merge:
        pending = pending_batches[batch_id]
        return jsonify({
            'result': 'pending_review',
            'batch_id': batch_id,
            'old_acc': pending.get('old_acc'),
            'new_acc': pending.get('new_acc'),
            'old_f1': pending.get('old_f1'),
            'new_f1': pending.get('new_f1'),
            'train_time_seconds': pending.get('train_time_seconds'),
            'processed_count': pending.get('processed_count'),
            'total_count': pending.get('total_count'),
            'errors': pending.get('errors'),
            'note': 'Batch pending admin review. Call /confirm_batch to finalize.'
        }), 200

    back_up = os.path.join(backup_dir, f'meta_fusion_{batch_id}.joblib')
    meta_model_path = os.path.join(models_dir, 'meta_fusion.joblib')
    feature_models_path = get_feature_models_path(models_dir)
    classes_path = os.path.join(models_dir, 'classes.txt')
    test_root = os.path.join(os.getcwd(), 'image')

    os.makedirs(backup_dir, exist_ok=True)

    # Create batch-specific backup
    shutil.copy(meta_model_path, back_up)

    try:
        old_acc, old_f1 = evaluate(test_root, model_dir=models_dir, size=100)
    except Exception as e:
        return jsonify({'error': f'Failed to evaluate old model: {str(e)}'}), 500

    processed_images: List[np.ndarray] = []
    processed_labels: List[int] = []
    errors: List[str] = []
    image_records: List[Dict[str, Any]] = []

    for idx, img_data in enumerate(images_data):
        try:
            img = None
            if img_data.get('image_base64'):
                img_data_b64 = img_data['image_base64']
                if img_data_b64.startswith('data:image'):
                    img_data_b64 = img_data_b64.split(',')[1]
                img_bytes = base64.b64decode(img_data_b64)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            elif img_data.get('image_url'):
                img_path = img_data['image_url']
                # Normalize dashboard-style /uploads/ paths
                if img_path.startswith('/uploads/'):
                    normalized = os.path.join(str(UPLOAD_DIR), img_path.lstrip('/uploads/'))
                    if os.path.exists(normalized):
                        img = cv2.imread(normalized, cv2.IMREAD_COLOR)
                if img is None:
                    if not os.path.isabs(img_path):
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        else:
                            data_dir = os.environ.get('DATA_DIR', '')
                            if data_dir:
                                full_path = os.path.join(data_dir, img_path.lstrip('/'))
                                if os.path.exists(full_path):
                                    img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                            if img is None:
                                upload_path = os.path.join(str(UPLOAD_DIR), img_path.lstrip('/uploads/'))
                                if os.path.exists(upload_path):
                                    img = cv2.imread(upload_path, cv2.IMREAD_COLOR)
                    else:
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is None:
                msg = f"Image {idx}: Cannot read image"
                errors.append(msg)
                image_records.append({
                    'index': idx,
                    'image_url': img_data.get('image_url'),
                    'status': 'error',
                    'reason': msg
                })
                continue

            fruit = (img_data.get('admin_fruit')
                     or img_data.get('user_fruit')
                     or img_data.get('fruit')
                     or img_data.get('predicted_fruit')
                     or '').strip()
            ripeness = (img_data.get('admin_ripeness')
                        or img_data.get('user_ripeness')
                        or img_data.get('ripeness')
                        or img_data.get('predicted_ripeness')
                        or '').strip()

            class_idx, class_name = convert_to_class_label(fruit, ripeness if ripeness else None)
            if class_idx is None:
                msg = f"Image {idx}: Invalid fruit/ripeness combination (fruit='{fruit}', ripeness='{ripeness}')"
                errors.append(msg)
                image_records.append({
                    'index': idx,
                    'image_url': img_data.get('image_url'),
                    'fruit': fruit,
                    'ripeness': ripeness,
                    'status': 'error',
                    'reason': msg
                })
                continue

            processed_images.append(img)
            processed_labels.append(class_idx)
            image_records.append({
                'index': idx,
                'image_url': img_data.get('image_url'),
                'fruit': fruit,
                'ripeness': ripeness,
                'status': 'pending'
            })
        except Exception as e:
            msg = f"Image {idx}: Error processing - {str(e)}"
            errors.append(msg)
            image_records.append({
                'index': idx,
                'image_url': img_data.get('image_url'),
                'status': 'error',
                'reason': msg
            })
            continue

    if len(processed_images) == 0:
        shutil.copy(back_up, meta_model_path)
        rejected_at = current_timestamp()
        batch_history.append({
            'batch_id': batch_id,
            'status': 'rejected',
            'rejected_at': rejected_at,
            'merged_at': None,
            'force': force_merge,
            'errors': errors,
            'images': image_records,
            'metrics_before': None,
            'metrics_after': None,
            'train_time_seconds': 0.0,
            'note': 'No valid images processed'
        })
        save_batch_history(batch_history)
        return jsonify({'error': 'No valid images processed', 'details': errors, 'batch_id': batch_id}), 400

    try:
        train_start = time.perf_counter()
        update_meta_model(
            meta_model_path,
            feature_models_path,
            classes_path,
            processed_images,
            processed_labels,
            size=100,
            alpha=0.1
        )
        train_time_seconds = round(time.perf_counter() - train_start, 4)
    except Exception as e:
        shutil.copy(back_up, meta_model_path)
        rejected_at = current_timestamp()
        batch_history.append({
            'batch_id': batch_id,
            'status': 'rejected',
            'rejected_at': rejected_at,
            'merged_at': None,
            'force': force_merge,
            'errors': errors + [f'Failed to update model: {str(e)}'],
            'images': image_records,
            'metrics_before': {'acc': old_acc, 'f1': old_f1},
            'metrics_after': None,
            'train_time_seconds': 0.0,
            'note': 'Model update failed'
        })
        save_batch_history(batch_history)
        return jsonify({'error': f'Failed to update model: {str(e)}', 'batch_id': batch_id}), 500

    try:
        new_acc, new_f1 = evaluate(test_root, model_dir=models_dir, size=100)
    except Exception as e:
        shutil.copy(back_up, meta_model_path)
        rejected_at = current_timestamp()
        batch_history.append({
            'batch_id': batch_id,
            'status': 'rejected',
            'rejected_at': rejected_at,
            'merged_at': None,
            'force': force_merge,
            'errors': errors + [f'Failed to evaluate new model: {str(e)}'],
            'images': image_records,
            'metrics_before': {'acc': old_acc, 'f1': old_f1},
            'metrics_after': None,
            'train_time_seconds': train_time_seconds,
            'note': 'Evaluation failed'
        })
        save_batch_history(batch_history)
        return jsonify({'error': f'Failed to evaluate new model: {str(e)}', 'batch_id': batch_id}), 500

    current_time = current_timestamp()
    
    # If not auto_decide, save to pending and return for admin review
    if not auto_decide:
        # Store pending batch info for later confirmation
        pending_batches = load_pending_batches()
        pending_batches[batch_id] = {
            'batch_id': batch_id,
            'backup_path': back_up,
            'old_acc': old_acc,
            'old_f1': old_f1,
            'new_acc': new_acc,
            'new_f1': new_f1,
            'train_time_seconds': train_time_seconds,
            'processed_count': len(processed_images),
            'total_count': len(images_data),
            'errors': errors if errors else [],
            'images': image_records,
            'created_at': current_time,
            'force': force_merge
        }
        save_pending_batches(pending_batches)
        
        return jsonify({
            'result': 'pending_review',
            'batch_id': batch_id,
            'old_acc': old_acc,
            'new_acc': new_acc,
            'old_f1': old_f1,
            'new_f1': new_f1,
            'processed_count': len(processed_images),
            'total_count': len(images_data),
            'errors': errors if errors else None,
            'train_time_seconds': train_time_seconds,
            'created_at': current_time,
            'note': 'Model trained. Awaiting admin decision. Call /confirm_batch to finalize.'
        }), 200

    # Legacy auto-decide mode
    if new_acc > old_acc or force_merge:
        for record in image_records:
            if record.get('status') == 'pending':
                record['status'] = 'merged'
        batch_entry = {
            'batch_id': batch_id,
            'status': 'merged',
            'merged_at': current_time,
            'rejected_at': None,
            'force': force_merge,
            'errors': errors if errors else [],
            'images': image_records,
            'metrics_before': {'acc': old_acc, 'f1': old_f1},
            'metrics_after': {'acc': new_acc, 'f1': new_f1},
            'train_time_seconds': train_time_seconds
        }
        batch_history.append(batch_entry)
        save_batch_history(batch_history)
        result = {
            'result': 'accepted',
            'batch_id': batch_id,
            'old_acc': old_acc,
            'new_acc': new_acc,
            'old_f1': old_f1,
            'new_f1': new_f1,
            'processed_count': len(processed_images),
            'total_count': len(images_data),
            'errors': errors if errors else None,
            'train_time_seconds': train_time_seconds,
            'merged_at': current_time
        }
        return jsonify(result), 200

    shutil.copy(back_up, meta_model_path)
    for record in image_records:
        if record.get('status') == 'pending':
            record['status'] = 'rejected'

    batch_entry = {
        'batch_id': batch_id,
        'status': 'rejected',
        'merged_at': None,
        'rejected_at': current_time,
        'force': force_merge,
        'errors': errors if errors else [],
        'images': image_records,
        'metrics_before': {'acc': old_acc, 'f1': old_f1},
        'metrics_after': {'acc': new_acc, 'f1': new_f1},
        'train_time_seconds': train_time_seconds,
        'note': 'Update rejected: new accuracy not better than old accuracy'
    }
    batch_history.append(batch_entry)
    save_batch_history(batch_history)

    result = {
        'result': 'rejected',
        'batch_id': batch_id,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'old_f1': old_f1,
        'new_f1': new_f1,
        'processed_count': len(processed_images),
        'total_count': len(images_data),
        'note': 'Update rejected: new accuracy not better than old accuracy',
        'errors': errors if errors else None,
        'train_time_seconds': train_time_seconds,
        'rejected_at': current_time
    }
    return jsonify(result), 200


@app.route('/confirm_batch/<batch_id>', methods=['POST'])
def confirm_batch(batch_id: str):
    """
    Confirm or reject a pending batch after admin review.
    
    Expected JSON:
    {
      "action": "accept" | "reject"
    }
    """
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()
    action = data.get('action', '').lower()
    
    if action not in ('accept', 'reject'):
        return jsonify({'error': '"action" must be "accept" or "reject"'}), 400

    pending_batches = load_pending_batches()
    
    if batch_id not in pending_batches:
        # Check if already finalized
        batch_history = load_batch_history()
        previous_entries = [entry for entry in batch_history if entry.get('batch_id') == batch_id]
        if previous_entries:
            last_entry = previous_entries[-1]
            return jsonify({
                'error': 'Batch already finalized',
                'batch_id': batch_id,
                'status': last_entry.get('status'),
                'merged_at': last_entry.get('merged_at'),
                'rejected_at': last_entry.get('rejected_at')
            }), 409
        return jsonify({'error': f'Pending batch {batch_id} not found'}), 404

    pending = pending_batches[batch_id]
    backup_path = pending.get('backup_path')
    meta_model_path = os.path.join(models_dir, 'meta_fusion.joblib')
    current_time = current_timestamp()
    
    batch_history = load_batch_history()
    image_records = pending.get('images', [])
    
    if action == 'accept':
        # Keep the new model (already in place after training)
        for record in image_records:
            if record.get('status') == 'pending':
                record['status'] = 'merged'
        
        batch_entry = {
            'batch_id': batch_id,
            'status': 'merged',
            'merged_at': current_time,
            'rejected_at': None,
            'force': pending.get('force', False),
            'errors': pending.get('errors', []),
            'images': image_records,
            'metrics_before': {'acc': pending['old_acc'], 'f1': pending['old_f1']},
            'metrics_after': {'acc': pending['new_acc'], 'f1': pending['new_f1']},
            'train_time_seconds': pending.get('train_time_seconds', 0)
        }
        batch_history.append(batch_entry)
        save_batch_history(batch_history)
        
        # Remove from pending
        del pending_batches[batch_id]
        save_pending_batches(pending_batches)
        
        # Clean up backup file
        if backup_path and os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except Exception:
                pass
        
        return jsonify({
            'result': 'accepted',
            'batch_id': batch_id,
            'old_acc': pending['old_acc'],
            'new_acc': pending['new_acc'],
            'old_f1': pending['old_f1'],
            'new_f1': pending['new_f1'],
            'train_time_seconds': pending.get('train_time_seconds'),
            'merged_at': current_time
        }), 200
    
    else:  # action == 'reject'
        # Restore the backup model
        if backup_path and os.path.exists(backup_path):
            shutil.copy(backup_path, meta_model_path)
            try:
                os.remove(backup_path)
            except Exception:
                pass
        
        for record in image_records:
            if record.get('status') == 'pending':
                record['status'] = 'rejected'
        
        batch_entry = {
            'batch_id': batch_id,
            'status': 'rejected',
            'merged_at': None,
            'rejected_at': current_time,
            'force': pending.get('force', False),
            'errors': pending.get('errors', []),
            'images': image_records,
            'metrics_before': {'acc': pending['old_acc'], 'f1': pending['old_f1']},
            'metrics_after': {'acc': pending['new_acc'], 'f1': pending['new_f1']},
            'train_time_seconds': pending.get('train_time_seconds', 0),
            'note': 'Rejected by admin'
        }
        batch_history.append(batch_entry)
        save_batch_history(batch_history)
        
        # Remove from pending
        del pending_batches[batch_id]
        save_pending_batches(pending_batches)
        
        return jsonify({
            'result': 'rejected',
            'batch_id': batch_id,
            'old_acc': pending['old_acc'],
            'new_acc': pending['new_acc'],
            'old_f1': pending['old_f1'],
            'new_f1': pending['new_f1'],
            'train_time_seconds': pending.get('train_time_seconds'),
            'rejected_at': current_time,
            'note': 'Rejected by admin'
        }), 200


@app.route('/pending_batches', methods=['GET'])
def get_pending_batches():
    """
    Fetch all batches pending admin review.
    Filters out any batches that are already finalized in batch_history.
    """
    pending_batches = load_pending_batches()
    batch_history = load_batch_history()
    
    # Get set of finalized batch IDs from history
    finalized_batch_ids = set(entry.get('batch_id') for entry in batch_history if entry.get('batch_id'))
    
    # Filter out any pending batches that are already finalized
    active_pending = {}
    stale_batch_ids = []
    for batch_id, batch_data in pending_batches.items():
        if batch_id not in finalized_batch_ids:
            active_pending[batch_id] = batch_data
        else:
            stale_batch_ids.append(batch_id)
    
    # Clean up stale entries from pending_batches if any found
    if stale_batch_ids:
        print(f"Cleaning up {len(stale_batch_ids)} stale pending batches: {stale_batch_ids}")
        save_pending_batches(active_pending)
    
    return jsonify({
        'batches': list(active_pending.values()),
        'count': len(active_pending)
    }), 200


@app.route('/batch_history', methods=['GET'])
def get_batch_history():
    """
    Fetch batch merge/reject history with optional filtering.
    Query parameters:
    - batch_id: Filter by specific batch ID
    - status: Filter by status ('merged' or 'rejected')
    - limit: Maximum number of results to return (default: 100)
    """
    batch_history = load_batch_history()
    
    # Filter by batch_id if provided
    batch_id = request.args.get('batch_id')
    if batch_id:
        batch_history = [entry for entry in batch_history if entry.get('batch_id') == batch_id]
    
    # Filter by status if provided
    status = request.args.get('status')
    if status:
        batch_history = [entry for entry in batch_history if entry.get('status') == status.lower()]
    
    # Sort by most recent first (merged_at or rejected_at)
    def get_timestamp(entry):
        return entry.get('merged_at') or entry.get('rejected_at') or ''
    batch_history.sort(key=get_timestamp, reverse=True)
    
    # Apply limit
    limit = request.args.get('limit', type=int, default=100)
    if limit > 0:
        batch_history = batch_history[:limit]
    
    # Calculate summary statistics
    total_batches = len(load_batch_history())
    merged_count = sum(1 for entry in load_batch_history() if entry.get('status') == 'merged')
    rejected_count = sum(1 for entry in load_batch_history() if entry.get('status') == 'rejected')
    
    return jsonify({
        'batches': batch_history,
        'summary': {
            'total': total_batches,
            'merged': merged_count,
            'rejected': rejected_count,
            'returned': len(batch_history)
        }
    }), 200


@app.route('/batch_history/<batch_id>', methods=['GET'])
def get_batch_by_id(batch_id):
    """
    Get detailed information about a specific batch by ID.
    Returns all history entries for this batch_id (in case of force merges).
    """
    batch_history = load_batch_history()
    entries = [entry for entry in batch_history if entry.get('batch_id') == batch_id]
    
    if not entries:
        return jsonify({'error': f'Batch {batch_id} not found'}), 404
    
    # Sort by most recent first
    def get_timestamp(entry):
        return entry.get('merged_at') or entry.get('rejected_at') or ''
    entries.sort(key=get_timestamp, reverse=True)
    
    return jsonify({
        'batch_id': batch_id,
        'entries': entries,
        'latest': entries[0] if entries else None,
        'total_entries': len(entries)
    }), 200


@app.route('/model_info', methods=['GET'])
def model_info():
    """Provide current model metadata for dashboard."""
    meta_model_path = os.path.join(models_dir, "meta_fusion.joblib")
    feature_models_path = get_feature_models_path()
    classes_path = os.path.join(models_dir, "classes.txt")

    info = {
        "version": os.environ.get("MODEL_VERSION", "unknown"),
        "last_updated": get_file_info(meta_model_path).get("modified"),
        "feature_models_path": feature_models_path,
        "meta_fusion_path": meta_model_path,
        "classes_path": classes_path,
        "meta_model": get_file_info(meta_model_path),
        "feature_models": get_file_info(feature_models_path),
        "classes": get_file_info(classes_path),
    }
    return jsonify(info), 200


# Initialize admin account on startup
init_admin_account()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)