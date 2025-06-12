import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def coin_measurement(image, coin_results):
    coin_diameter_cm = 2.7  # diameter koin sebenarnya (dalam cm)

    if not coin_results or len(coin_results[0].boxes.xywh) == 0:
        print("No coin results found")
        return None, image

    # PILIH kotak dengan ukuran TERKECIL (asumsi itu adalah koin)
    boxes = coin_results[0].boxes.xywh
    areas = [w * h for x, y, w, h in boxes]
    smallest_idx = np.argmin(areas)
    x_coin, y_coin, w_coin, h_coin = boxes[smallest_idx]

    # Crop dan deteksi lingkaran (opsional, bisa abaikan jika YOLO sudah cukup)
    x1 = int(x_coin - w_coin / 2)
    y1 = int(y_coin - h_coin / 2)
    x2 = int(x_coin + w_coin / 2)
    y2 = int(y_coin + h_coin / 2)
    crop_coin = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop_coin, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    detected_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=100,
    )

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        a, b, r = detected_circles[0, 0]
        coin_diameter_px = 2 * r
        cv2.circle(image, (x1 + a, y1 + b), r, (255, 0, 0), 2)
    else:
        print("No circles detected — fallback to box size.")
        coin_diameter_px = min(w_coin, h_coin)

    scale_factor = coin_diameter_cm / coin_diameter_px
    print(f"✅ Coin diameter in pixels: {coin_diameter_px}")
    print(f"📏 Scale factor: {scale_factor:.5f}")
    return scale_factor, image


def baby_measurement(image, scale_factor, pose_model):
    # Deteksi pose
    pose_results = pose_model(image, stream=False)
    image1 = pose_results[0].plot()

    # Ambil keypoints dari hasil deteksi
    keypoints = pose_results[0].keypoints.xy[0]

    # Ambil titik yang dibutuhkan: 0 = nose, 5 = left shoulder, 15 = left ankle
    indices = np.array([0, 5, 15])
    selected_keypoints = keypoints[indices]

    # Assign ke variabel terpisah
    nose, left_shoulder, left_ankle = selected_keypoints

    # Cek apakah titik valid (tidak kosong/NaN)
    if any(np.isnan(k).any() for k in [nose, left_shoulder, left_ankle]):
        print("❌ Keypoints tidak terdeteksi dengan lengkap.")
        return None, image1

    # Hitung panjang lurus dari hidung ke pergelangan kaki kiri
    length_px = np.linalg.norm(left_ankle - nose)

    print(f"✅ Baby length in pixels: {length_px:.2f}")
    
    # Konversi ke centimeter
    baby_length_cm = length_px * scale_factor
    print(f"📏 Perkiraan panjang bayi: {baby_length_cm:.2f} cm")
    
    return baby_length_cm, image1


def download_image(url_or_path):
    try:
        if url_or_path.startswith("http"):
            response = requests.get(url_or_path)
            response.raise_for_status()
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
        else:
            img = cv2.imread(url_or_path)

        if img is None:
            print("Image could not be loaded.")
            return None

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[-1] != 3:
            print(f"Unsupported image format with {img.shape[-1]} channels.")
            return None
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
def measure_all(url):
    pose_model = YOLO("keypoints/yolo11s-pose.pt")
    coin_model = YOLO("coin/yolo11s.pt")
    image = download_image(url)

    if image is None or len(image.shape) != 3 or image.shape[-1] != 3:
        print("Invalid image format. Ensure the image is RGB or convertible to RGB.")
        return None

    coin_results = coin_model(image, stream=False)
    if not coin_results:
        print("No coin detected.")
        return None

    scale_factor, image_with_coin = coin_measurement(image, coin_results)
    baby_length, final_image = baby_measurement(image_with_coin, scale_factor, pose_model)

    # ✅ Simpan hasil ke file
    cv2.imwrite("output.jpg", final_image)
    print("✅ Gambar hasil disimpan sebagai 'output.jpg'.")

    # ✅ Tampilkan hasil
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Panjang bayi: {baby_length:.2f} cm")
    plt.axis("off")
    plt.show()

    return baby_length
