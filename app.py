from flask import Flask, Response, render_template_string
import cv2
import requests
from ultralytics import YOLO
import time
import nest_asyncio
import os

nest_asyncio.apply()
app = Flask(__name__)

# تحميل النموذج
model = YOLO("best.pt")  # لازم ترفع الملف ده مع المشروع

# ترجمة الفئات
label_translation = {
    "Sick": "مريضة",
    "Dead": "ميتة",
    "Healthy": "سليمة"
}

api_url = "http://farmsmanagement.runasp.net/api/Notifiactions/CreateNotification"
last_label = None
last_sent_time = 0

def generate_frames():
    global last_label, last_sent_time
    ip_camera_url = "http://<Tailscale_or_public_IP>:8080/video"  # عدّلي هنا
    cap = cv2.VideoCapture(ip_camera_url)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.resize(frame, (640, 480))

        try:
            results = model(frame, verbose=False)
        except Exception as e:
            print(f"❌ YOLO Error: {e}")
            continue

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                conf = float(scores[i])
                if conf < 0.75:
                    continue

                x1, y1, x2, y2 = boxes[i].astype(int)
                class_id = int(classes[i])
                label = model.names[class_id]
                arabic_label = label_translation.get(label, label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{arabic_label} ({conf:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                if label in ["Sick", "Dead"]:
                    if (label != last_label or time.time() - last_sent_time > 10):
                        data = {
                            "body": f"فرخة {arabic_label}",
                            "userId": 24,
                            "barnId": 3,
                            "isRead": False
                        }
                        try:
                            requests.post(api_url, json=data)
                            print(f"🚨 إشعار: {data['body']}")
                            last_label = label
                            last_sent_time = time.time()
                        except Exception as e:
                            print(f"❌ خطأ إرسال الإشعار: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head><title>بث مباشر مع كشف الدواجن</title></head>
        <body>
            <h1>🐔 بث مباشر لكشف حالة الدواجن</h1>
            <img src="{{ url_for('video') }}" width="80%">
        </body>
        </html>
    ''')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
