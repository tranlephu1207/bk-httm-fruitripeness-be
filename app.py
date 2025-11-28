from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flasgger import Swagger
from joblib import load
from src_ripeness.features import extract_all
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import cv2
import json
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'change-this-in-production')
app.config['SWAGGER'] = {'uiversion': 3}

jwt = JWTManager(app)
swagger = Swagger(app)
CORS(app, resources={r"/*": {"origins": "*"}})

models_dir = "model_ripeness"
base_models = load(f"{models_dir}/feature_svms.joblib")
meta_clf = load(f"{models_dir}/meta_fusion.joblib")
with open(f"{models_dir}/classes.txt") as f:
    classes = [line.strip() for line in f if line.strip()]

DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))
UPLOAD_DIR = Path(os.environ.get('UPLOAD_DIR', DATA_DIR / 'uploads'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FRUITS_FILE = DATA_DIR / 'fruits.json'
SCANS_FILE = DATA_DIR / 'scans.json'
ORDERS_FILE = DATA_DIR / 'orders.json'
USERS_FILE = DATA_DIR / 'users.json'

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
{% if result %}
  <h3>Result:</h3>
  <b>Fruit:</b> {{ result['fruit'] }}<br>
  <b>Ripeness:</b> {{ result['ripeness'] }}<br>
  <b>Score:</b> {{ result['score'] }}
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

    overall_accuracy = (user_agreed / total) if total else 0

    daily = [
        {
            'date': date,
            'total': stats['total'],
            'correct': stats['agreed'],
            'incorrect': stats['disagreed'],
            'accuracy': round((stats['agreed'] / stats['total']) * 100, 2) if stats['total'] else 0
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


# Initialize admin account on startup
init_admin_account()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)