from flask import Flask, request, jsonify, render_template
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import zipfile
import tempfile
from werkzeug.utils import secure_filename
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

def map_to_real_or_fake(label):
    real_labels = [
        "10note", "20note", "50note",
        "1Hundrednote", "2Hundrednote",
        "5Hundrednote", "2Thousandnote"
    ]
    return "Real" if label.strip().lower() in [r.lower() for r in real_labels] else "Fake"

def get_prediction_features(is_real):
    return [
        {"name": "Gandhi Portrait Alignment", "valid": is_real},
        {"name": "Color Shift Pattern", "valid": is_real},
        {"name": "Security Thread Visibility", "valid": is_real},
        {"name": "Serial Number Format", "valid": is_real},
        {"name": "Note Texture & Grain", "valid": is_real}
    ]

def predict(image_path):
    from model import VGG16FineTuned, load_model_and_classes
    model, classes = load_model_and_classes(device)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    label = classes[predicted.item()]
    return map_to_real_or_fake(label), label

@app.route('/predict_single', methods=['POST'])
def predict_single():
    file = request.files.get('imagefile')
    if not file or file.filename == '':
        return jsonify({'error': 'No image uploaded'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    result, label = predict(file_path)
    features = get_prediction_features(result == "Real")

    return render_template('result.html',
                           result=result,
                           label=label,
                           image_file=filename,
                           features=features)

@app.route('/predict_zip', methods=['POST'])
def predict_zip():
    file = request.files.get('zipfile')
    if not file:
        return jsonify({'status': 'error', 'message': 'No zip file uploaded'}), 400

    zip_filename = secure_filename(file.filename)
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(root, fname)
                    result, label = predict(fpath)

                    new_path = os.path.join(UPLOAD_FOLDER, fname)
                    Image.open(fpath).save(new_path)

                    results.append({
                        'filename': fname,
                        'result': result,
                        'label': label,
                        'features': get_prediction_features(result == "Real")
                    })

    return render_template('zip_result.html',
                           results=results,
                           zip_filename=zip_filename)

@app.route('/test_accuracy')
def test_accuracy():
    from model import VGG16FineTuned, load_model_and_classes
    model, _ = load_model_and_classes(device)

    test_dir = 'C:/Users/rohan/Documents/rishi/LY project/Final project/dataset/Test'
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return f"<h1>Testing Accuracy: {accuracy:.2f}%</h1>"

@app.route('/confusion_matrix')
def confusion_matrix_route():
    from model import load_model_and_classes
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64

    model, classes = load_model_and_classes(device)

    test_dir = 'C:/Users/rohan/Documents/rishi/LY project/Final project/dataset/Test'
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    real_labels = [
        "10note", "20note", "50note",
        "1Hundrednote", "2Hundrednote",
        "5Hundrednote", "2Thousandnote"
    ]
    real_labels_lower = [label.lower() for label in real_labels]

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for true_idx, pred_idx in zip(labels, preds.cpu()):
                true_class = classes[true_idx]
                pred_class = classes[pred_idx]

                true_binary = "Real" if true_class.lower() in real_labels_lower else "Fake"
                pred_binary = "Real" if pred_class.lower() in real_labels_lower else "Fake"

                all_labels.append(true_binary)
                all_preds.append(pred_binary)

    cm = confusion_matrix(all_labels, all_preds, labels=["Fake", "Real"])
    report = classification_report(all_labels, all_preds, labels=["Fake", "Real"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Real vs Fake)')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return render_template('confusion_matrix.html', image_base64=image_base64, report=report)

if __name__ == '__main__':
    app.run(debug=True)
