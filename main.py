from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from io import BytesIO
from homography import stitch_images

app = Flask(__name__, static_folder='.', static_url_path='')


@app.get("/")
def root():
    return app.send_static_file("index.html")


@app.post("/api/homography")
def api_homography():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "file1 and file2 are required"}), 400
    f1 = request.files['file1'].read()
    f2 = request.files['file2'].read()
    img1 = cv2.imdecode(np.frombuffer(f1, dtype=np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(f2, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return jsonify({"error": "Invalid image(s) provided"}), 400
    try:
        stitched = stitch_images(img1, img2)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    ok, buf = cv2.imencode('.png', stitched)
    if not ok:
        return jsonify({"error": "Failed to encode image"}), 500
    return send_file(BytesIO(buf.tobytes()), mimetype='image/png')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
