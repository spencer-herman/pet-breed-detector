from flask import Flask, request, render_template, send_file
import os
from PIL import Image, ImageDraw
import io
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

#Get the API key from the environment variable
API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError("Please set the ROBOFLOW_API_KEY environment variable")

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

@app.route('/')
def index():
    return render_template('upload_form.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']

    # Save the uploaded image to a temporary file
    temp_image_path = "temp_image.jpg"
    file.save(temp_image_path)

    # Perform inference using the temporary image path
    result = CLIENT.infer(temp_image_path, model_id='dog_breed_detector/1')

    # Open the image with PIL for further processing
    image = Image.open(temp_image_path)

    # Draw the bounding box and labels on the image
    draw = ImageDraw.Draw(image)
    predictions = result['predictions']

    for pred in predictions:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        label = pred['class']
        confidence = pred['confidence']
        box = [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)]
        draw.rectangle(box, outline="red", width=3)
        text = f"{label}: {confidence:.2f}"
        draw.text((x - width / 2, y - height / 2 - 10), text, fill="red")

    # Save the result image to bytes
    result_img_io = io.BytesIO()
    image.save(result_img_io, 'JPEG')
    result_img_io.seek(0)

    # Remove the temporary image file after use
    os.remove(temp_image_path)

    # Return the image with the predictions as a response
    return send_file(result_img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)