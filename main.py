from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from io import BytesIO
from homography import stitch_images

app = Flask(__name__, static_folder='.', static_url_path='')


def resize_max_edge(img: np.ndarray, max_edge: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale >= 0.999:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
        # downscale for speed, reduce features and iterations
        img1_s = resize_max_edge(img1, 1600)
        img2_s = resize_max_edge(img2, 1600)
        stitched = stitch_images(img1_s, img2_s, iterations=4000, threshold=3.0, nfeatures=2000)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    ok, buf = cv2.imencode('.png', stitched)
    if not ok:
        return jsonify({"error": "Failed to encode image"}), 500
    return send_file(BytesIO(buf.tobytes()), mimetype='image/png')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
