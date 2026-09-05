import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import fitz as pymupdf
import ollama
from ultralytics import YOLO

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    MODEL_PATH,
    DPI,
    CONF_THRESHOLD,
    LABELS,
    HTR_LABELS,
    OCR_MODEL,
    OCR_PROMPT,
    COLUMN_COLORS_HEX,
    COLUMN_COLORS_RGBA,
    CSV_COLUMNS,
    INVERT_TRANSFORM,
)


# Crée l'arborescence des dossiers 

def ensure_directories() -> dict[str, Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages_dir = OUTPUT_DIR / "pages"
    thumbs_dir = OUTPUT_DIR / "thumbs"
    crops_dir = OUTPUT_DIR / "crops"
    bbox_dir = OUTPUT_DIR / "bbox"

    for d in (OUTPUT_DIR, pages_dir, thumbs_dir, crops_dir, bbox_dir):
        d.mkdir(parents=True, exist_ok=True)

    for label_name in LABELS.values():
        (crops_dir / label_name).mkdir(parents=True, exist_ok=True)

    return {
        "pages": pages_dir,
        "thumbs": thumbs_dir,
        "crops": crops_dir,
        "bbox": bbox_dir,
    }


# Sauvegarde les fichiers d'entrée, extrait les pages en HD et génère des miniatures plus légère pour ne pas ralentir le processus.

def save_and_split_documents(files, dirs: dict[str, Path]) -> list[tuple[str, Path]]:
    prepared = []
    pages_dir = dirs["pages"]
    thumbs_dir = dirs["thumbs"]

    for f in files:
        src_path = Path(f.name if hasattr(f, "name") else f)
        dest_input = INPUT_DIR / src_path.name
        dest_input.write_bytes(src_path.read_bytes())

        if dest_input.suffix.lower() == ".pdf":
            doc = pymupdf.open(dest_input)
            matrix = pymupdf.Matrix(DPI / 72, DPI / 72)
            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=matrix)
                img_path = pages_dir / f"{dest_input.stem}_page{i:03d}.png"
                pix.save(img_path)

                thumb_path = thumbs_dir / f"{img_path.stem}_thumb.jpg"
                with Image.open(img_path) as img:
                    img.thumbnail((600, 800))
                    img.convert("RGB").save(thumb_path, "JPEG", quality=75)

                prepared.append((img_path.stem, img_path))
            doc.close()
        else:
            img_path = pages_dir / dest_input.name
            img_path.write_bytes(dest_input.read_bytes())

            thumb_path = thumbs_dir / f"{img_path.stem}_thumb.jpg"
            with Image.open(img_path) as img:
                img.thumbnail((600, 800))
                img.convert("RGB").save(thumb_path, "JPEG", quality=75)

            prepared.append((img_path.stem, img_path))

    return prepared


# Bloc HTML pour la légende des 9 colonnes du tableau.

def generate_legend_html() -> str:
    labels = [
        "1. Numéro",
        "2. Objet",
        "3. Description",
        "4. Illustration",
        "5. N° Photo",
        "6. N° Dessin",
        "7. Localisation",
        "8. Période",
        "9. Destination",
    ]
    html = "<div style='display: flex; flex-direction: column; gap: 4px; font-size: 13px;'>"
    for idx, (label, color) in enumerate(zip(labels, COLUMN_COLORS_HEX)):
        html += (
            f"<div style='display: flex; align-items: center; gap: 8px;'>"
            f"<div style='width: 16px; height: 16px; background-color: {color}; "
            f"border: 1px solid #333; border-radius: 3px;'></div>"
            f"<span><b>Col {idx+1}</b> : {label}</span>"
            f"</div>"
        )
    html += "</div>"
    return html


# Dessine les lignes de calibrage pour les colonnes

def draw_calibration_lines(image: Image.Image, clicked_x: list[float]) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    h = img_copy.height

    for idx, x in enumerate(clicked_x):
        color = COLUMN_COLORS_HEX[idx % len(COLUMN_COLORS_HEX)]
        draw.line([(x, 0), (x, h)], fill=color, width=3)

    return img_copy


# Génère l'aperçu calibré une seule fois pour éviter la latence entre deux clics

def generate_and_cache_preview(stem: str, dirs: dict[str, Path], ratios: list[float]) -> Path:
    thumb_path = dirs["thumbs"] / f"{stem}_thumb.jpg"
    preview_path = dirs["thumbs"] / f"{stem}_preview.jpg"

    if not thumb_path.exists():
        thumb_path = dirs["pages"] / f"{stem}.png"

    img = Image.open(thumb_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    curr_x = 0.0

    for col_i, r in enumerate(ratios):
        col_w = w * r
        color = COLUMN_COLORS_RGBA[col_i % len(COLUMN_COLORS_RGBA)]
        draw.rectangle([curr_x, 0, curr_x + col_w, h], fill=color)
        draw.line([(curr_x, 0), (curr_x, h)], fill=(0, 0, 0, 180), width=2)
        curr_x += col_w

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    combined.save(preview_path, "JPEG", quality=80)
    return preview_path


# Applique l'inversion d'image albumentation et l'OCR via Ollama.

def perform_ocr(crop_img: Image.Image) -> str:
    inverted_arr = INVERT_TRANSFORM(image=np.array(crop_img))["image"]
    inverted_img = Image.fromarray(inverted_arr)

    buffer = io.BytesIO()
    inverted_img.save(buffer, format="PNG")

    try:
        res = ollama.generate(
            model=OCR_MODEL,
            prompt=OCR_PROMPT,
            images=[buffer.getvalue()]
        )
        return res.get("response", "").strip()
    except Exception as e:
        return f"[Erreur OCR: {e}]"


# Détecte les blocs via YOLO et lance l'HTR

def run_yolo_and_htr(
    prepared_pages: list[tuple[str, Path]],
    page_ratios: dict[str, list[float]],
    default_ratios: list[float],
    dirs: dict[str, Path]
) -> dict:
    model = YOLO(MODEL_PATH)
    all_bboxes = {}

    crops_dir = dirs["crops"]
    bbox_dir = dirs["bbox"]

    for stem, img_path in prepared_pages:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        ratios = page_ratios.get(stem, default_ratios)
        if not ratios:
            ratios = [1.0 / 9.0] * 9

        col_boundaries = [0.0]
        acc = 0.0
        for r in ratios:
            acc += r * w
            col_boundaries.append(acc)

        results = model.predict(source=str(img_path), conf=CONF_THRESHOLD)
        bboxes_list = []

        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        box_count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = LABELS.get(cls_id, "inconnu")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                xmin, ymin, xmax, ymax = map(int, xyxy)
                x_center = (xmin + xmax) / 2.0

                col_idx = 0
                for c_i in range(len(col_boundaries) - 1):
                    if col_boundaries[c_i] <= x_center < col_boundaries[c_i + 1]:
                        col_idx = c_i
                        break

                box_count += 1
                crop = img.crop((xmin, ymin, xmax, ymax))

                text_recognized = ""
                crop_rel_path = ""

                if label in HTR_LABELS:
                    text_recognized = perform_ocr(crop)
                else:
                    crop_filename = f"{stem}_box{box_count:03d}_{label}.png"
                    save_p = crops_dir / label / crop_filename
                    crop.save(save_p)
                    crop_rel_path = str(save_p.relative_to(OUTPUT_DIR))

                color = COLUMN_COLORS_HEX[col_idx % len(COLUMN_COLORS_HEX)]
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                draw.text((xmin, max(0, ymin - 12)), f"{label} ({col_idx+1})", fill=color)

                bboxes_list.append({
                    "id": box_count,
                    "label": label,
                    "confidence": conf,
                    "box": [xmin, ymin, xmax, ymax],
                    "column": col_idx + 1,
                    "text": text_recognized,
                    "crop_path": crop_rel_path
                })

        for x_b in col_boundaries[1:-1]:
            draw.line([(x_b, 0), (x_b, h)], fill="black", width=2)

        annotated_img.save(bbox_dir / f"{stem}_bbox.png")
        bboxes_list.sort(key=lambda b: (b["box"][1], b["box"][0]))
        all_bboxes[stem] = bboxes_list

    json_path = bbox_dir / "all_detections.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_bboxes, f, ensure_ascii=False, indent=2)

    return all_bboxes


# Structure des résultats OCR détectés dans un DataFrame Pandas et création du CSV.

def build_csv_output(all_bboxes: dict, dirs: dict[str, Path]) -> tuple[Path, pd.DataFrame]:
    rows = []

    for stem, bboxes in all_bboxes.items():
        col1_boxes = [b for b in bboxes if b["column"] == 1]

        if not col1_boxes:
            row_data = {col: "" for col in CSV_COLUMNS}
            row_data["Page"] = stem
            for b in bboxes:
                col_name = CSV_COLUMNS[b["column"] - 1]
                content = b["text"] if b["text"] else b["crop_path"]
                row_data[col_name] = (row_data[col_name] + " " + content).strip()
            rows.append(row_data)
        else:
            for i, num_box in enumerate(col1_boxes):
                y_min = num_box["box"][1]
                y_max = col1_boxes[i + 1]["box"][1] if i + 1 < len(col1_boxes) else float("inf")

                row_data = {col: "" for col in CSV_COLUMNS}
                row_data["Page"] = stem

                for b in bboxes:
                    box_y = (b["box"][1] + b["box"][3]) / 2.0
                    if y_min <= box_y < y_max:
                        col_name = CSV_COLUMNS[b["column"] - 1]
                        content = b["text"] if b["text"] else b["crop_path"]
                        row_data[col_name] = (row_data[col_name] + " " + content).strip()

                rows.append(row_data)

    df = pd.DataFrame(rows)
    csv_path = dirs["bbox"] / "resultats_ocr.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return csv_path, df