
# # krtik's work
# now no need
# import cv2
# import time
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load face detector and model
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# classifier = load_model('emotion_model.h5')

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# def get_final_emotion(readings):
#     """Aggregate readings into final classification with custom logic"""
#     total = len(readings)
#     counts = {e: readings.count(e) for e in set(readings)}
#     neutral_count = counts.get("Neutral", 0)
#     positive_count = counts.get("Happy", 0) + counts.get("Surprise", 0)
#     negative_count = counts.get("Angry", 0) + counts.get("Fear", 0) + counts.get("Sad", 0)

#     neutral_ratio = neutral_count / total if total else 0

#     # Confusion if neutral is too high
#     if neutral_ratio >= 0.65:  # confusion threshold
#         return "Confusion"
#     elif 0.50 <= neutral_ratio < 0.65:  # still mostly neutral
#         return "Neutral"
#     elif positive_count > negative_count:
#         return "Positive"
#     elif negative_count > positive_count:
#         return "Negative"
#     else:
#         return "Neutral"

# def main():
#     cap = cv2.VideoCapture(0)
#     readings = []
#     duration = 120  # 2 minutes
#     start_time = time.time()
#     samples_needed = 30
#     interval = duration / samples_needed  # ~4 seconds interval
#     last_capture = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#         current_time = time.time()
#         if faces is not None and len(faces) > 0:
#             for (x, y, w, h) in faces:
#                 roi_gray = gray[y:y+h, x:x+w]
#                 roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)

#                 preds = classifier.predict(roi, verbose=0)[0]
#                 emotion = emotion_labels[np.argmax(preds)]
#                 score = np.max(preds)

#                 # --- Custom rule: treat Sad with low confidence as Neutral ---
#                 if emotion == "Sad" and score <= 0.5:
#                     emotion = "Neutral"

#                 if current_time - last_capture >= interval and len(readings) < samples_needed:
#                     readings.append(emotion)
#                     last_capture = current_time
#                     print(f"[{len(readings)}/{samples_needed}] Detected: {emotion} ({score:.2f})")

#         # Break if 2 minutes passed or enough samples collected
#         if (current_time - start_time >= duration) or (len(readings) >= samples_needed):
#             break

#         # Optional: show live feed
#         cv2.imshow("Emotion Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     final_emotion = get_final_emotion(readings)
#     print("\n============================")
#     print(f"Final Classified Emotion: {final_emotion}")
#     print("============================")

# if __name__ == "__main__":
#    main() 


# working code
import cv2
import time
from collections import Counter
from fer import FER
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== Emotion Classification Rules ==========
def classify_emotion(readings):
    counts = Counter(readings)
    total = sum(counts.values())

    if total == 0:
        return "neutral"

    perc = {emo: counts[emo] / total * 100 for emo in counts}
    neutral_pct = perc.get("neutral", 0)

    # Rule 1: Confusion if neutral >= 45%
    if neutral_pct >= 45:
        return "confusion"

    # Rule 2: Positive (happy ≥ 25%)
    if perc.get("happy", 0) >= 25:
        return "positive"

    # Rule 3: Negative (sad ≥ 15% or angry/disgust/fear ≥ 15%)
    if (perc.get("sad", 0) >= 15 or
        perc.get("angry", 0) >= 15 or
        perc.get("disgust", 0) >= 15 or
        perc.get("fear", 0) >= 15):
        return "negative"

    return "neutral"


# ========== Emotion Detection Logic ==========
def detect_emotion_task(result_container):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        result_container["emotion"] = "neutral"
        return

    readings = []
    duration = 60   # 2 minutes
    interval = 2     # ~30 readings
    end_time = time.time() + duration

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            readings.append("neutral")
            break

        results = detector.detect_emotions(frame)

        if results:
            emotions = results[0]["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            score = emotions[top_emotion]

            if top_emotion == "sad" and score < 0.5:
                top_emotion = "neutral"

            if top_emotion == "neutral" and score >= 0.7:
                top_emotion = "happy"

            readings.append(top_emotion)
            print(f"Captured: {top_emotion} ({score:.2f})")
        else:
            readings.append("neutral")

        cv2.imshow("Emotion Logger", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(interval)

    cap.release()
    cv2.destroyAllWindows()

    final_state = classify_emotion(readings)
    print(f"✅ Final Emotion: {final_state}")
    result_container["emotion"] = final_state


# ========== Flask Routes ==========
@app.route("/start-emotion", methods=["POST"])
def start_emotion():
    result_container = {}
    thread = threading.Thread(target=detect_emotion_task, args=(result_container,))
    thread.start()
    thread.join()  # Wait until finished

    return jsonify({
        "status": "completed",
        "final_emotion": result_container.get("emotion", "neutral")
    })


@app.route("/")
def index():
    return "🎥 Emotion Detection Flask Server Running!"


if __name__ == "__main__":
    app.run(port=8000, debug=True)


