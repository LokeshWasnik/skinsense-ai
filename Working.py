# skin_cancer_app.py

# -------------------- IMPORTS --------------------
import os
import numpy as np
import time
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array


# -------------------- FLASK SETUP --------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/image'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['IMAGE_UPLOADS'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24).hex()

# -------------------- STEP 1: DATA LOADING --------------------
def load_data():
    dataset_path = os.environ.get("DATASET_PATH", r'D:\images\sorted_skin_images')
    img_height, img_width = 28, 28
    batch_size = 32

    full_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_size = int(0.25 * len(full_ds))
    train_ds = full_ds.skip(val_size)
    val_ds = full_ds.take(val_size)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds

# -------------------- STEP 2: MODEL BUILD --------------------
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,3)),
        MaxPooling2D(), BatchNormalization(),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(), BatchNormalization(),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(), BatchNormalization(),

        Flatten(), Dropout(0.3),
        Dense(256, activation='relu'), BatchNormalization(),
        Dense(128, activation='relu'), BatchNormalization(),
        Dense(64, activation='relu'), BatchNormalization(),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=Adamax(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------- TRAIN MODEL IF NOT EXISTS --------------------
model_path = 'SkinCancerClassificationModelFinal.h5'

if not os.path.exists(model_path):
    print("Training model...")
    train_ds, val_ds = load_data()
    model = create_model()
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=1e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[lr_scheduler])
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation Accuracy: {val_acc:.2f}")
    model.save(model_path)
    print("Model trained and saved!")
else:
    print("Model already exists. Skipping training...")

# -------------------- LOAD MODEL --------------------
model = load_model(model_path)


# -------------------- CLASS LABELS & INFO --------------------
cancer_classes = {
    0: 'Actinic keratoses and intraepithelial carcinomae',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic Nevus',
    5: 'Vascular Lesion',
    6: 'Melanoma',
}

cancer_info = {
    'Actinic keratoses and intraepithelial carcinomae': "Precancerous, rough, scaly patches on sun-exposed skin. Early treatment can prevent progression to squamous cell carcinoma.",
    'Basal Cell Carcinoma': "The most common skin cancer. Grows slowly and rarely spreads, but can cause local damage if untreated.",
    'Benign Keratosis': "Non-cancerous skin growths, often appearing as warty or waxy spots. Generally harmless but can resemble skin cancer.",
    'Dermatofibroma': "A benign, firm bump often found on the legs. Not dangerous and usually does not require treatment.",
    'Melanocytic Nevus': "Commonly known as a mole. Most are harmless, but changes in size, color, or shape should be checked by a doctor.",
    'Vascular Lesion': "Abnormal blood vessels in the skin, usually benign. Some types may require monitoring or treatment.",
    'Melanoma': "A serious form of skin cancer that can spread rapidly. Early detection and treatment are critical for a good outcome. See a dermatologist promptly if suspected.",
}

# -------------------- FLASK ROUTES --------------------
@app.route('/')
def home():
    return render_template('forms/index.html')

@app.route('/q2', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        uploaded_image = request.files.get('file')
        if not uploaded_image:
            return "No file uploaded", 400

        filename = uploaded_image.filename
        filepath = os.path.join(app.config['IMAGE_UPLOADS'], filename)
        uploaded_image.save(filepath)

        img = load_img(filepath, target_size=(28, 28))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        label = cancer_classes.get(pred_class, "Unknown")
        info = cancer_info.get(label, "No additional information available.")

        # Optional: auto-delete uploaded image after prediction
        # time.sleep(2)
        # os.remove(filepath)

        return render_template('forms/q2_new.html', filename=filename, diagnosis=label, cancer_info=info)
    return render_template('forms/q2_new.html')

@app.route('/index/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='image/' + filename), code=301)

@app.route('/learn')
def learn():
    return render_template('learn.html')  # You need to create this HTML file

@app.route('/try_now')
def try_now():
    return redirect(url_for('classify_image'))

# -------------------- RUN FLASK APP --------------------
if __name__ == '__main__':
    print("Flask app is starting on http://localhost:2500")
    import webbrowser
    import os
    # Only open browser if not running in the Flask reloader subprocess
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        webbrowser.open("http://localhost:2500")
    app.run(host='0.0.0.0', port=2500, debug=True)
