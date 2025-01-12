from flask import Flask, render_template, Response
import cv2
import pickle
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

# Path to the directory containing models
directory_path = r"C:\\Users\\vidjo\\Downloads\\ASL_Translator-main\\ASL_Translator-main\\data\\landmark_data"

models = []

# Load models from all subdirectories
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"Found file: {file_path}")
            if filename.endswith('.pkl'):
                try:
                    with open(file_path, 'rb') as file:
                        model_dict = pickle.load(file)
                    models.append(model_dict['model'])
                    print(f"Successfully loaded model from: {file_path}")
                except Exception as e:
                    print(f"Error loading file '{file_path}': {e}")

else:
    print(f"Error: Directory '{directory_path}' does not exist.")
    exit(1)

# Check if any models were loaded
if not models:
    print("Error: No valid models were loaded.")
    exit(1)

print(f"Loaded {len(models)} model(s).")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Initialize webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    # Use the first loaded model as the default
    default_model = models[0]

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract landmark coordinates
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Ensure the input shape matches the model's requirement
            if len(data_aux) == 84:  # Check correct input length
                prediction = default_model.predict([np.asarray(data_aux)])
                confidence = np.max(default_model.predict_proba([np.asarray(data_aux)]))

                if confidence >= 0.5:
                    predicted_character = prediction[0]
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shehacksvideo1.html')
def shehack_video():
    return render_template('shehacksvideo1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Release resources when shutting down
@app.teardown_appcontext
def cleanup(exception=None):
    cap.release()
    hands.close()

if __name__ == '__main__':
    app.run(debug=True)
