from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, storage, db
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import datetime

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    print("Serving the home page.")
    predicted_label, error = classify_image()
    if error:
        # Render an error page or return an error message
        return f'<html><body><h1>Error</h1><p>{error}</p></body></html>'
    else:
        # Render a page with the classification result
        return (
            f'<html><body><h1>Hello, World!</h1><p>Welcome to Adelia school server.</p><br><br>'
            f'<h1>Classification Result</h1><p>{predicted_label}</p></body></html>')


# Initialize Firebase Admin SDK
cred = credentials.Certificate('adelia-recycle-ai-firebase-adminsdk-eo4gg-1e9b563855.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'adelia-recycle-ai.appspot.com',
    'databaseURL': 'https://adelia-recycle-ai-default-rtdb.firebaseio.com/'
})
print("Firebase Admin initialized.")

bucket = storage.bucket()

scheduler = BackgroundScheduler(daemon=True)
scheduler.start()

# Ensure the scheduler stops when the app exits
atexit.register(lambda: scheduler.shutdown())

last_processed_time = None


def is_image_new(image_path):
    global last_processed_time
    blob = bucket.blob(image_path)
    blob.reload()
    current_time = blob.updated  # `updated` is a datetime object representing the last modification time

    if not last_processed_time or current_time > last_processed_time:
        last_processed_time = current_time
        return True
    return False


def check_and_classify_image():
    # Implement logic to determine if the image is new since the last check.
    # This could involve comparing timestamps or checking a marker in the database.
    # For simplicity, assume you have a function `is_image_new()` that returns True if the image is new.

    image_path = get_latest_image_path()
    if image_path and is_image_new(image_path):
        print("Found a new image, classifying...")
        classify_image()
    else:
        print("No new image found or the image is the same as last checked.")


# Schedule the `check_and_classify_image` to run every 2 seconds
scheduler.add_job(check_and_classify_image, 'interval', seconds=2)


def download_image(image_path):
    print(f"Downloading image from Firebase Storage: {image_path}")
    blob = bucket.blob(image_path)
    local_path = f'tmp/{os.path.basename(image_path)}'
    blob.download_to_filename(local_path)
    print(f"Image downloaded to: {local_path}")
    return local_path


def update_classification_result(image_path, label):
    print(f"Updating classification result in Firebase Realtime Database for {image_path} with label {label}")
    result_ref = db.reference(f'classification_results/')
    result_ref.set({'label': label, 'updated_at': datetime.datetime.utcnow().isoformat() + 'Z'})
    print("Database updated.")


# Load the TFLite model and labels
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
print("TFLite model loaded.")

with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
print("Labels loaded.")


def process_image(image_path, target_size=(224, 224)):
    print("Processing image.")
    # Open the image file directly, ensuring it's in the correct mode
    with Image.open(image_path).convert("RGB") as image:
        # Resize the image to the target size
        image = image.resize(target_size)
        # Convert the image to a numpy array
        image_array = np.array(image, dtype=np.uint8)
        # Add a batch dimension (TensorFlow expects this)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array


def get_latest_image_path():
    # List all objects in the bucket
    blobs = list(bucket.list_blobs())
    if blobs:
        # Assuming the latest uploaded file is the image you want to process
        latest_blob = blobs[-1]
        print(f"Latest image found: {latest_blob.name}")
        return latest_blob.name
    else:
        print("No images found in Firebase Storage.")
        return None


def classify_image():
    image_path = get_latest_image_path()
    if not image_path:
        return None, 'No image found in Firebase Storage'

    try:
        local_image_path = download_image(image_path)
        processed_image = process_image(local_image_path)

        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = labels[np.argmax(output_data)]

        print(f"label result is: {predicted_label}...................")

        # Optionally, delete the downloaded image if no longer needed
        os.remove(local_image_path)

        # Update Firebase Realtime Database with the classification result
        update_classification_result(image_path, predicted_label)

        return predicted_label, None  # Return label and no error

    except Exception as e:
        return None, str(e)  # Return no label and the error message


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=False, host='0.0.0.0')
