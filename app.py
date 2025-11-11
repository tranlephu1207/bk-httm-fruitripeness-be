from flask import Flask, request, jsonify, render_template_string
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from flasgger import Swagger
import cv2
import numpy as np
from joblib import load
from src_ripeness.features import extract_all
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
# JWT configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'change-this-in-production')
jwt = JWTManager(app)
app.config['SWAGGER'] = {'uiversion': 3}  # enables Swagger UI 3
swagger = Swagger(app)
CORS(app, resources={r"/*": {"origins": "*"}})
models_dir = "model_ripeness"
base_models = load(f"{models_dir}/feature_svms.joblib")
meta_clf = load(f"{models_dir}/meta_fusion.joblib")
with open(f"{models_dir}/classes.txt") as f:
    classes = [line.strip() for line in f if line.strip()]

def parse_label(label):
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

HTML_FORM = """
<!doctype html>
<title>Fruit Ripeness Classifier</title>
<h2>Upload an image to classify ripeness</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=image>
  <input type=submit value=Upload>
</form>
{% if result %}
  <h3>Result:</h3>
  <b>Fruit:</b> {{ result['fruit'] }}<br>
  <b>Ripeness:</b> {{ result['ripeness'] }}<br>
  <b>Score:</b> {{ result['score'] }}
{% endif %}
"""

# --- Simple JSON User Store ---
USERS_FILE = 'users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        import json
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    import json
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)


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
        if 'image' not in request.files:
            result = {'error': 'No image uploaded'}
        else:
            file = request.files['image']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                result = {'error': 'Cannot read image'}
            else:
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
                label = classes[cidx]
                ripeness, fruit = parse_label(label)
                result = {
                    'fruit': fruit,
                    'ripeness': ripeness,
                    'score': round(cscore, 4)
                }
    return render_template_string(HTML_FORM, result=result)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict top-1 fruit ripeness
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: Image file to classify
    responses:
      200:
        description: Top-1 prediction
        schema:
          type: object
          properties:
            fruit:
              type: string
            ripeness:
              type: string
            score:
              type: number
              format: float
      400:
        description: Bad request (no image or unreadable)
    """
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
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
    label = classes[cidx]
    ripeness, fruit = parse_label(label)
    response = {
        'fruit': fruit,
        'ripeness': ripeness,
        'score': round(cscore, 4)
    }
    return jsonify(response)


@app.route('/predict_topk', methods=['POST'])
def predict_topk():
    """Predict top-5 classes with probabilities
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: Image file to classify
    responses:
      200:
        description: Top-5 predictions
        schema:
          type: object
          properties:
            topk:
              type: array
              items:
                type: object
                properties:
                  label:
                    type: string
                  fruit:
                    type: string
                  ripeness:
                    type: string
                  score:
                    type: number
                    format: float
      400:
        description: Bad request (no image or unreadable)
    """
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'No image uploaded'}), 400
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Cannot read image'}), 400
    img_resized = cv2.resize(img, (100, 100))
    feats = extract_all(img_resized)
    probs = []
    for m, f in zip(base_models, feats):
        prob = m.predict_proba([f])[0]
        probs.append(prob)
    meta_input = np.concatenate(probs).reshape(1, -1)
    fused = meta_clf.predict_proba(meta_input)[0]

    k = 5
    top_indices = np.argsort(fused)[::-1][:k]
    results = []
    for idx in top_indices:
        label = classes[int(idx)]
        ripeness, fruit = parse_label(label)
        results.append({
            'label': label,
            'fruit': fruit,
            'ripeness': ripeness,
            'score': round(float(fused[int(idx)]), 4)
        })

    return jsonify({'topk': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)