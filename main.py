import os
import warnings
import cv2
import easyocr
import imutils
from flask import Flask, render_template, request

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Папка для сохранения загруженных и обработанных картинок
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Инициализация EasyOCR при старте сервера
reader = easyocr.Reader(['en', 'ru'])


def process_plate_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Не удалось прочитать изображение"

    img = imutils.resize(img, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plate_cascade = cv2.CascadeClassifier(
        'haarcascade_russian_plate_number.xml'
    )
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        maxSize=(400, 200),
    )

    detected_number = "Номер не найден"

    for x, y, w, h in plates:
        padding = 5
        img_h, img_w = img.shape[:2]
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(img_w, x + w + padding), min(img_h, y + h + padding)

        plate_crop = img[y1:y2, x1:x2]
        results = reader.readtext(plate_crop)

        parsed_text = ""
        for bbox, text, prob in results:
            if prob > 0.2:
                parsed_text += text + " "

        clean_text = "".join(c for c in parsed_text if c.isalnum()).upper()
        if clean_text:
            detected_number = clean_text

        # Отрисовка результатов
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            clean_text,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # Сохраняем обработанный кадр в static/uploads
    cv2.imwrite(save_path, img)
    return detected_number, None


@app.route('/', methods=['GET', 'POST'])
def index():
    plate_number = None
    image_url = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'Файл не найден'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'Файл не выбран'
            else:
                input_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'input.jpg'
                )
                output_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'result.jpg'
                )

                file.save(input_path)

                # Запуск детекции и OCR
                plate_number, err = process_plate_image(
                    input_path, output_path
                )
                if err:
                    error = err
                else:
                    # Путь к изображению для отображения на HTML-странице
                    image_url = f"/{output_path.replace(os.sep, '/')}"

    return render_template(
        'index.html',
        plate_number=plate_number,
        image_url=image_url,
        error=error,
    )


if __name__ == '__main__':
    app.run(debug=True)