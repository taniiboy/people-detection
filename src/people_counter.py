from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict, deque
import cv2
import time
import csv
import re

#Configurations
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "src" / "models"
models = [
    MODEL_DIR / "yolov8n.pt",
    MODEL_DIR / "yolov8s.pt",
    MODEL_DIR / "yolov8m.pt",
    MODEL_DIR / "yolov8x.pt",
    MODEL_DIR / "yolo11n.pt",
    MODEL_DIR / "yolo11s.pt",
    MODEL_DIR / "yolo11m.pt",
    MODEL_DIR / "yolo11x.pt",
    MODEL_DIR / "yolo26n.pt",
    MODEL_DIR / "yolo26s.pt",
    MODEL_DIR / "yolo26m.pt",
    MODEL_DIR / "yolo26x.pt",
]
VIDEO_PATH = BASE_DIR / "input_videos" / "video4.mp4"
OUTPUT_DIR = BASE_DIR / "output" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "comparison_results.csv"

CONFIDENCE = 0.2
PERSON_CLASS_ID = 0

LINE_Y = 100

MIN_MOVEMENT = 3

DRAW_TRACK_HISTORY = True
TRACK_HISTORY_LENGTH = 8

PREPROCESSING_MODE = "clahe_sharpen"
# mögliche Werte:
# "none"
# "brightness_contrast"
# "strong_contrast"
# "clahe"
# "sharpen"
# "clahe_sharpen"
# "strong_clahe_sharpen"

TRACKER = "bytetrack.yaml"
# Alternative:botsort.yaml

SHOW_PREVIEW = True
SAVE_YOLO_VIDEO = True
SAVE_PROCESSED_VIDEO = True
GROUND_TRUTH_ENTRY = 33
GROUND_TRUTH_EXIT = 15

#Functions
def preprocess_frame(frame, mode="none"):
    """
    Bearbeitet einen Frame vor der YOLO-Erkennung.
    Das Originalvideo bleibt unverändert.
    """

    if mode == "none":
        return frame

    if mode == "brightness_contrast":
        alpha = 1.25
        beta = 20
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if mode == "strong_contrast":
        alpha = 1.6
        beta = 35
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if mode == "clahe":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    if mode == "sharpen":
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)

        return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

    if mode == "clahe_sharpen":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced_frame, (0, 0), 1.0)

        return cv2.addWeighted(enhanced_frame, 1.4, blurred, -0.4, 0)

    if mode == "strong_clahe_sharpen":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(enhanced_frame, (0, 0), 1.0)

        return cv2.addWeighted(enhanced_frame, 1.8, blurred, -0.8, 0)

    raise ValueError(f"Unbekannter Preprocessing-Modus: {mode}")


def parse_model_info(model_name):
    """
    Extrahiert Modellfamilie und Modellgröße.
    Beispiele:
    yolov8n.pt  -> family = YOLOv8, size = n
    yolo11s.pt  -> family = YOLO11, size = s
    yolo26x.pt  -> family = YOLO26, size = x
    """
    stem = Path(model_name).stem
    match = re.match(r"(yolov?\d+)([nslmx])", stem)

    if match:
        family = match.group(1).upper().replace("YOLOV", "YOLOv")
        size = match.group(2)
    else:
        family = stem
        size = "unknown"
    return family, size

def draw_info(annotated_frame, entry_count, exit_count, line_y):
    """
    Zeichnet Zähllinie und Text auf das Bild.
    """

    height, width = annotated_frame.shape[:2]
    cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
    cv2.putText(
        annotated_frame,
        "Counting Line",
        (20, max(line_y - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Entry: {entry_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Exit: {exit_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

def draw_track_history(annotated_frame, track_history, active_ids):
    """
    Zeichnet kurze Bewegungsspuren der aktuell sichtbaren Personen.
    """

    if not DRAW_TRACK_HISTORY:
        return

    for track_id in active_ids:
        points = track_history.get(track_id)
        if points is None or len(points) < 2:
            continue
        point_list = list(points)
        for i in range(1, len(point_list)):
            x1, y1 = point_list[i - 1]
            x2, y2 = point_list[i]
            cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

def calculate_error(predicted, ground_truth):
    return abs(predicted - ground_truth)

def process_video_with_model(model_name):
    print("\n" + "=" * 60)
    print(f"Teste Modell: {model_name}")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video wurde nicht gefunden: {VIDEO_PATH}")

    family, size = parse_model_info(model_name)
    try:
        model = YOLO(str(model_name))
    except Exception as e:
        print(f"Modell konnte nicht geladen werden: {model_name}")
        print(f"Fehler: {e}")
        return {
            "model": Path(model_name).name,
            "family": family,
            "size": size,
            "status": "load_error",
            "preprocessing": PREPROCESSING_MODE,
            "confidence": CONFIDENCE,
            "tracker": TRACKER,
            "line_y": LINE_Y,
            "frames": 0,
            "time_seconds": None,
            "fps_processing": None,
            "entry_count": None,
            "exit_count": None,
            "entry_error": None,
            "exit_error": None,
            "total_error": None,
            "yolo_video": None,
            "processed_video": None,
        }

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {VIDEO_PATH}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Videopfad:", VIDEO_PATH)
    print("Video geöffnet:", cap.isOpened())
    print("Video FPS:", original_fps)
    print("Auflösung:", frame_width, "x", frame_height)
    print("Preprocessing:", PREPROCESSING_MODE)
    print("Confidence:", CONFIDENCE)
    print("Tracker:", TRACKER)

    entry_count = 0
    exit_count = 0
    previous_positions = {}
    counted_ids = set()
    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))
    frame_count = 0
    start_time = time.time()
    yolo_writer = None
    processed_writer = None
    yolo_video_path = None
    processed_video_path = None
    video_fps = original_fps if original_fps > 0 else 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    model_stem = Path(model_name).stem
    conf_label = str(CONFIDENCE).replace(".", "")

    if SAVE_YOLO_VIDEO:
        yolo_video_path = OUTPUT_DIR / (f"{model_stem}_{PREPROCESSING_MODE}_conf{conf_label}_yolo.mp4")
        yolo_writer = cv2.VideoWriter(str(yolo_video_path), fourcc, video_fps, (frame_width, frame_height))

    if SAVE_PROCESSED_VIDEO:
        processed_video_path = OUTPUT_DIR / (f"{model_stem}_{PREPROCESSING_MODE}_conf{conf_label}_processed.mp4")
        processed_writer = cv2.VideoWriter(str(processed_video_path), fourcc, video_fps, (frame_width, frame_height))

    if SHOW_PREVIEW:
        cv2.namedWindow(f"YOLO People Counter - {model_name}", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        processed_frame = preprocess_frame(frame, PREPROCESSING_MODE)
        results = model.track(
            processed_frame,
            persist=True,
            conf=CONFIDENCE,
            classes=[PERSON_CLASS_ID],
            tracker=TRACKER,
            verbose=False,
        )

        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        active_ids = set()

        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            active_ids = set(ids)

            for box, track_id in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                track_history[track_id].append((center_x, center_y))
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(
                    annotated_frame,
                    f"ID {track_id}",
                    (center_x + 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

                if track_id in previous_positions:
                    previous_y = previous_positions[track_id]
                    movement = center_y - previous_y

                    if track_id not in counted_ids:
                        # Von oben nach unten
                        if (previous_y < LINE_Y and center_y >= LINE_Y and movement >= MIN_MOVEMENT):
                            exit_count += 1
                            counted_ids.add(track_id)

                        # Von unten nach oben
                        elif (previous_y > LINE_Y and center_y <= LINE_Y and movement <= -MIN_MOVEMENT):
                            entry_count += 1
                            counted_ids.add(track_id)

                previous_positions[track_id] = center_y

        draw_track_history(annotated_frame, track_history, active_ids)
        draw_info(annotated_frame, entry_count, exit_count, LINE_Y)

        if yolo_writer is not None:
            yolo_writer.write(annotated_frame)

        if processed_writer is not None:
            processed_writer.write(processed_frame)

        if SHOW_PREVIEW:
            cv2.imshow(f"YOLO People Counter - {model_name}", annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            # q beendet nur aktuelles Modell
            if key == ord("q"):
                print("Aktuelles Modell wurde durch Nutzer abgebrochen.")
                break

            # ESC beendet komplett
            if key == 27:
                print("Programm wurde durch Nutzer beendet.")
                cap.release()
                if yolo_writer is not None:
                    yolo_writer.release()

                if processed_writer is not None:
                    processed_writer.release()
                cv2.destroyAllWindows()
                raise KeyboardInterrupt

    end_time = time.time()
    total_time = end_time - start_time

    cap.release()

    if yolo_writer is not None:
        yolo_writer.release()
    if processed_writer is not None:
        processed_writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    processing_fps = frame_count / total_time if total_time > 0 else 0
    entry_error = calculate_error(entry_count, GROUND_TRUTH_ENTRY)
    exit_error = calculate_error(exit_count, GROUND_TRUTH_EXIT)
    total_error = entry_error + exit_error
    print(f"Frames verarbeitet: {frame_count}")
    print(f"Laufzeit: {total_time:.2f} Sekunden")
    print(f"Processing FPS: {processing_fps:.2f}")
    print(f"Entry: {entry_count}")
    print(f"Exit: {exit_count}")
    print(f"Entry Fehler: {entry_error}")
    print(f"Exit Fehler: {exit_error}")
    print(f"Gesamtfehler: {total_error}")

    if yolo_video_path is not None:
        print("YOLO-Video gespeichert unter:", yolo_video_path)

    if processed_video_path is not None:
        print("Processed-Video gespeichert unter:", processed_video_path)

    return {
        "model": Path(model_name).name,
        "family": family,
        "size": size,
        "status": "ok",
        "preprocessing": PREPROCESSING_MODE,
        "confidence": CONFIDENCE,
        "tracker": TRACKER,
        "line_y": LINE_Y,
        "frames": frame_count,
        "time_seconds": round(total_time, 2),
        "fps_processing": round(processing_fps, 2),
        "entry_count": entry_count,
        "exit_count": exit_count,
        "entry_error": entry_error,
        "exit_error": exit_error,
        "total_error": total_error,
        "yolo_video": str(yolo_video_path) if yolo_video_path else None,
        "processed_video": str(processed_video_path) if processed_video_path else None,
    }

def save_results_to_csv(results):
    fieldnames = [
        "model",
        "family",
        "size",
        "status",
        "preprocessing",
        "confidence",
        "tracker",
        "line_y",
        "frames",
        "time_seconds",
        "fps_processing",
        "entry_count",
        "exit_count",
        "entry_error",
        "exit_error",
        "total_error",
        "yolo_video",
        "processed_video",
    ]

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nCSV gespeichert unter:")
    print(CSV_PATH)

def main():
    print("Starte YOLO-Modellvergleich")
    print("Video:", VIDEO_PATH)
    print("Existiert:", VIDEO_PATH.exists())
    print("Preprocessing:", PREPROCESSING_MODE)
    print("Confidence:", CONFIDENCE)
    print("Line Y:", LINE_Y)

    all_results = []

    for model_name in models:
        result = process_video_with_model(model_name)
        all_results.append(result)

    save_results_to_csv(all_results)

    print("\n" + "=" * 60)
    print("Zusammenfassung")
    print("=" * 60)

    for result in all_results:
        print(
            f"{result['model']}: "
            f"Status={result['status']}, "
            f"Family={result['family']}, "
            f"Size={result['size']}, "
            f"FPS={result['fps_processing']}, "
            f"Entry={result['entry_count']}, "
            f"Exit={result['exit_count']}, "
            f"Gesamtfehler={result['total_error']}"
        )

if __name__ == "__main__":
    main()
