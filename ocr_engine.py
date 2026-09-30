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


# Crée l'arborescence des dossiers de l'application.
# sous-dossiers crops sont créés automatiquement à partir des labels YOLO.


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


# Sauvegarde des fichiers d'entrée, extrait les pages en HD et génère
# des miniatures plus légères pour ne pas ralentir le processus.
# copie des fichiers dans le dossier INPUT_DIR pour conservation avant traitement

# Si le fichier est un PDF découpage ==> PNG avec 72 DPI

# Création miniature JPEG pour alléger et faciliter l'affichage

# si le fichier n'est pas un PDF ==> directement intégré dans le dossier des pages.
# retourne la liste des pages préparées avec leur nom et leur chemin.

def save_and_split_documents(
    files,
    dirs: dict[str, Path]
) -> list[tuple[str, Path]]:

    prepared = []
    pages_dir = dirs["pages"]
    thumbs_dir = dirs["thumbs"]

    for f in files:
        src_path = Path(f.name if hasattr(f, "name") else f)
        dest_input = INPUT_DIR / src_path.name


        dest_input.write_bytes(src_path.read_bytes())

        # Découpage des pages du PDF et génération des thumbnails.
        # Matrice pour augmenter la résolution pour le YOLO
        if dest_input.suffix.lower() == ".pdf":
            doc = pymupdf.open(dest_input)
            matrix = pymupdf.Matrix(DPI / 72, DPI / 72)

            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=matrix)

                img_path = pages_dir / (
                    f"{dest_input.stem}_page{i:03d}.png"
                )

                pix.save(img_path)

                # miniature  600x800 pixels pour la rapidité
                thumb_path = thumbs_dir / (
                    f"{img_path.stem}_thumb.jpg"
                )

                with Image.open(img_path) as img:
                    img.thumbnail((600, 800))
                    img.convert("RGB").save(
                        thumb_path,
                        "JPEG",
                        quality=75
                    )

                prepared.append((img_path.stem, img_path))

            doc.close()

        # Prise en compte des autres formats
        else:
            img_path = pages_dir / dest_input.name
            img_path.write_bytes(dest_input.read_bytes())

           # Création de la miniature
            thumb_path = thumbs_dir / (
                f"{img_path.stem}_thumb.jpg"
            )

            with Image.open(img_path) as img:
                img.thumbnail((600, 800))
                img.convert("RGB").save(
                    thumb_path,
                    "JPEG",
                    quality=75
                )

            prepared.append((img_path.stem, img_path))

    return prepared


#  HTML utilisé pour afficher la légende des colonnes à placer (n°; nom et couleur)
# retourne le bloc HTML complet pour être afficher dans l'interface

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

    html = (
        "<div style='display: flex; flex-direction: column; "
        "gap: 4px; font-size: 13px;'>"
    )

    for idx, (label, color) in enumerate(
        zip(labels, COLUMN_COLORS_HEX)
    ):
        html += (
            f"<div style='display: flex; align-items: center; gap: 8px;'>"
            f"<div style='width: 16px; height: 16px; "
            f"background-color: {color}; "
            f"border: 1px solid #333; border-radius: 3px;'></div>"
            f"<span><b>Col {idx+1}</b> : {label}</span>"
            f"</div>"
        )

    html += "</div>"

    return html


# Dessine les lignes de calibrage pour les colonnes.
#  copie de l'image pour ne pas la modifier 
#  valeur  clicked_x = une position horizontale

# lignes sur la hauteur de l'image.
# retourne la copie de l'image avec les lignes de calibration.

def draw_calibration_lines(
    image: Image.Image,
    clicked_x: list[float]
) -> Image.Image:

    # Copie de l'image 
    img_copy = image.copy()

    # outil de dessin
    draw = ImageDraw.Draw(img_copy)

    h = img_copy.height

    # boucle sur less positions X pour dessiner
    # une ligne verticale correspondant à chaque séparation de colonne.
    for idx, x in enumerate(clicked_x):
        color = COLUMN_COLORS_HEX[
            idx % len(COLUMN_COLORS_HEX)
        ]

        draw.line(
            [(x, 0), (x, h)],
            fill=color,
            width=3
        )

    return img_copy


# Apercu du découpage, une seule fois pour éviter la latence entre les clics

# récupère d'abord la miniature de la page.
# Si elle n'est pastrouvée, utilisation de l'original 

#  ratios pour déterminer la largeur de chaque colonne par rapport
# à la largeur totale de l'image.

# couche transparente pour ne pas modifier l'image

# Chaque colonne est représentée par un rectangle coloré et une ligne noire
# permettant de visualiser précisément sa séparation.
#retourne chemin de la prévisualisation de ce découpage

def generate_and_cache_preview(
    stem: str,
    dirs: dict[str, Path],
    ratios: list[float]
) -> Path:

    thumb_path = dirs["thumbs"] / f"{stem}_thumb.jpg"
    preview_path = dirs["thumbs"] / f"{stem}_preview.jpg"

    # Si la miniature n'existe pas, on utilise directement
    # l'image originale de la page.
    if not thumb_path.exists():
        thumb_path = dirs["pages"] / f"{stem}.png"

    img = Image.open(thumb_path).convert("RGBA")

    # couche transparente 
    # Les rectangles des colonnes seront dessinés sur cette couche
    # avant d'être fusionnés avec l'image originale.
    overlay = Image.new(
        "RGBA",
        img.size,
        (255, 255, 255, 0)
    )

    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    curr_x = 0.0

    # Calcul de la taille de chaque colonne dessinée pour chaque ratio.
    for col_i, r in enumerate(ratios):
        col_w = w * r

        #  couleur correspondant à la colonne.
        color = COLUMN_COLORS_RGBA[
            col_i % len(COLUMN_COLORS_RGBA)
        ]

        # Application de la couleur
        draw.rectangle(
            [curr_x, 0, curr_x + col_w, h],
            fill=color
        )

        #  ligne noire pour visualiser la séparation
        draw.line(
            [(curr_x, 0), (curr_x, h)],
            fill=(0, 0, 0, 180),
            width=2
        )

        curr_x += col_w

    # Fusion image originale avec la couche contenant
    # les différentes colonnes colorées.
    combined = Image.alpha_composite(
        img,
        overlay
    ).convert("RGB")

    # Sauvegarde de la prévisualisation 
    combined.save(
        preview_path,
        "JPEG",
        quality=80
    )

    return preview_path


# inversion d'image albumentation et l'HTR via Ollama.

# L'image du crop convertie en tableau NumPy pour transformation d'inversion ensuite reconvertie en image PIL.

# déplacé dans le buffer mémoire au format PNG afin d'éviter
# de créer un fichier temporaire sur le disque.

# L'image envoyée au modèle HTR configuré dans Ollama
# avec le prompt 

# retourne uniquement le texte reconnu par le modèle.

# En cas d'erreur pendant le traitement HTR ==> retourne message d'erreur et ppoursuit le traitement

def perform_ocr(crop_img: Image.Image) -> str:

    # Inversion des couleurs
    inverted_arr = INVERT_TRANSFORM(
        image=np.array(crop_img)
    )["image"]

    # Conversion du tableau NumPy transformé en image PIL
    inverted_img = Image.fromarray(inverted_arr)

    # buffer
    buffer = io.BytesIO()

    inverted_img.save(
        buffer,
        format="PNG"
    )

    try:
        # Envoi de l'image au modèle HTR via Ollama.
        res = ollama.generate(
            model=OCR_MODEL,
            prompt=OCR_PROMPT,
            images=[buffer.getvalue()]
        )

        # Récupération de la réponse du modèle.
        return res.get(
            "response",
            ""
        ).strip()

    # Gestion des erreurs 
    except Exception as e:
        return f"[Erreur OCR: {e}]"


# Détecte les blocs via YOLO et HTR.

# on récupère le last.pt entraîné

# ratios de colonnes utilisés pour calculer les limites horizontales
# des 9 colonnes du tableau.

# Pour chaque bounding box détectée, la fonction récupère le label,
# la confiance et les coordonnées de la détection.
# Le centre horizontal de chaque bounding box permet de déterminer
# dans quelle colonne l'élément se trouve.
# Les éléments appartenant aux labels HTR sont envoyés à la fonction OCR.
# Les autres éléments sont sauvegardés sous forme de crops dans leur dossier.
# Une image annotée est créée afin de visualiser les détections YOLO,
# les labels et les colonnes correspondantes.


# informations des détections sont enregistrées dans un JSON.
# retourne finalement toutes les bounding boxes détectées.

def run_yolo_and_htr(
    prepared_pages: list[tuple[str, Path]],
    page_ratios: dict[str, list[float]],
    default_ratios: list[float],
    dirs: dict[str, Path]
) -> dict:

    # chargement du modèle YOLO
    model = YOLO(MODEL_PATH)

    # stockage des résultats de toutes les pages
    all_bboxes = {}

    crops_dir = dirs["crops"]
    bbox_dir = dirs["bbox"]

    # parcours des pages préparées
    for stem, img_path in prepared_pages:
        img = Image.open(img_path).convert("RGB")

        w, h = img.size

        # récupération des ratios de la page
        # si aucun ratio enregistré ==> utilisation des ratios par défaut
        ratios = page_ratios.get(
            stem,
            default_ratios
        )

        # si aucun ratio ==> division de l'image en 9 colonnes égales
        if not ratios:
            ratios = [1.0 / 9.0] * 9

        # calcul des limites de chaque colonne en pixels
        # les ratios permettent de passer de proportions à des coordonnées X
        col_boundaries = [0.0]
        acc = 0.0

        for r in ratios:
            acc += r * w
            col_boundaries.append(acc)

        # lancement de YOLO sur la page
        # le seuil de confiance de base ( pourrait être modifié pour améliorer la détection ? )
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD
        )

        bboxes_list = []

        # copie de l'image pour dessiner les résultats
        # l'image originale reste inchangée
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        box_count = 0

        # parcours des résultats YOLO
        for r in results:
            for box in r.boxes:

                # récupération de la classe détectée
                cls_id = int(
                    box.cls[0].item()
                )

                # conversion de la classe en label
                # si le label n'existe pas ==> inconnu
                label = LABELS.get(
                    cls_id,
                    "inconnu"
                )

                # récupération du niveau de confiance
                conf = float(
                    box.conf[0].item()
                )

                # coordonnées de la bounding box
                xyxy = box.xyxy[0].tolist()

                xmin, ymin, xmax, ymax = map(
                    int,
                    xyxy
                )

                # calcul du centre de la box
                # permet de savoir dans quelle colonne se trouve la détection
                x_center = (
                    xmin + xmax
                ) / 2.0

                col_idx = 0

                # recherche de la colonne correspondant au centre de la box
                for c_i in range(
                    len(col_boundaries) - 1
                ):
                    if (
                        col_boundaries[c_i]
                        <= x_center
                        < col_boundaries[c_i + 1]
                    ):
                        col_idx = c_i
                        break

                box_count += 1

                # découpage de la zone détectée
                crop = img.crop(
                    (
                        xmin,
                        ymin,
                        xmax,
                        ymax
                    )
                )

                text_recognized = ""
                crop_rel_path = ""

                # si le label demande de l'HTR ==> envoie le crop à l'OCR
                if label in HTR_LABELS:
                    text_recognized = perform_ocr(
                        crop
                    )

                # sinon ==> sauvegarde du crop dans le dossier du label
                else:
                    crop_filename = (
                        f"{stem}_box{box_count:03d}_{label}.png"
                    )

                    save_p = (
                        crops_dir
                        / label
                        / crop_filename
                    )

                    crop.save(save_p)

                    # récupération du chemin relatif pour le JSON et le CSV
                    crop_rel_path = str(
                        save_p.relative_to(
                            OUTPUT_DIR
                        )
                    )

                # récupération de la couleur de la colonne
                color = COLUMN_COLORS_HEX[
                    col_idx % len(COLUMN_COLORS_HEX)
                ]

                # dessin de la bounding box
                draw.rectangle(
                    [
                        xmin,
                        ymin,
                        xmax,
                        ymax
                    ],
                    outline=color,
                    width=3
                )

                # affichage du label et du numéro de colonne
                draw.text(
                    (
                        xmin,
                        max(0, ymin - 12)
                    ),
                    f"{label} ({col_idx+1})",
                    fill=color
                )

                # stockage des infos de la détection
                # ces données seront utilisés ensuite pour le JSON et le CSV
                bboxes_list.append({
                    "id": box_count,
                    "label": label,
                    "confidence": conf,
                    "box": [
                        xmin,
                        ymin,
                        xmax,
                        ymax
                    ],
                    "column": col_idx + 1,
                    "text": text_recognized,
                    "crop_path": crop_rel_path
                })

        # dessin des séparations entre les colonnes
        # les bords de l'image ne sont pas concernés
        for x_b in col_boundaries[1:-1]:
            draw.line(
                [
                    (x_b, 0),
                    (x_b, h)
                ],
                fill="black",
                width=2
            )

        # sauvegarde de l'image avec les détections
        annotated_img.save(
            bbox_dir / f"{stem}_bbox.png"
        )

        # tri des boxes de haut en bas et de gauche à droite
        # permet de garder un ordre cohérent pour le CSV
        bboxes_list.sort(
            key=lambda b: (
                b["box"][1],
                b["box"][0]
            )
        )

        all_bboxes[stem] = bboxes_list

    # sauvegarde de toutes les détections dans un JSON
    # permet de garder les résultats détaillés du traitement
    json_path = bbox_dir / "all_detections.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_bboxes,
            f,
            ensure_ascii=False,
            indent=2
        )

    return all_bboxes


#  stockage des détection en DataFrame et création du CSV

# la colonne 1 = base pour retrouver les différentes lignes du tableau

# si aucun numéro n'est détecté ==> tous les éléments de la page sont regroupés
# si des numéros sont détectés ==> chaque numéro devient le début d'une ligne

# position verticale des boxes permet de savoir à quelle ligne elles appartiennent

# le texte OCR est utilisé quand il existe
# sinon ==> utilisation du chemin du crop pour faire le lien avec le dossier des illustrations pour le rendu final

# plusieurs éléments dans une même colonne sont regroupés dans la même cellule
# retourne le chemin du CSV et le DataFrame

def build_csv_output(
    all_bboxes: dict,
    dirs: dict[str, Path]
) -> tuple[Path, pd.DataFrame]:

    rows = []

    # parcours de toutes les pages
    for stem, bboxes in all_bboxes.items():

        # récupération des boxes de la colonne 1
        col1_boxes = [
            b
            for b in bboxes
            if b["column"] == 1
        ]

        # si aucun numéro détecté ==> une seule ligne pour toute la page
        if not col1_boxes:

            # création d'une ligne vide avec toutes les colonnes
            row_data = {
                col: ""
                for col in CSV_COLUMNS
            }

            row_data["Page"] = stem

            # parcours des détections de la page
            for b in bboxes:

                # récupération du nom de la colonne
                col_name = CSV_COLUMNS[
                    b["column"] - 1
                ]

                # texte OCR si disponible
                # sinon ==> chemin du crop
                content = (
                    b["text"]
                    if b["text"]
                    else b["crop_path"]
                )

                # ajout du contenu dans la colonne
                # si plusieurs boxes sont présentes ==> elles sont concaténées
                row_data[col_name] = (
                    row_data[col_name]
                    + " "
                    + content
                ).strip()

            rows.append(row_data)

        # si des numéros sont détectés ==> création d'une ligne par numéro
        else:
            for i, num_box in enumerate(col1_boxes):

                # début de la ligne avec la position du numéro
                y_min = num_box["box"][1]

                # fin de la ligne avec la position du numéro suivant
                # pour le dernier numéro ==> on prend toute la suite de la page
                y_max = (
                    col1_boxes[i + 1]["box"][1]
                    if i + 1 < len(col1_boxes)
                    else float("inf")
                )

                # création d'une ligne vide
                row_data = {
                    col: ""
                    for col in CSV_COLUMNS
                }

                row_data["Page"] = stem

                # parcours de toutes les détections
                for b in bboxes:

                    # calcul du centre vertical de la box
                    # permet de savoir si elle appartient à la ligne actuelle
                    box_y = (
                        b["box"][1]
                        + b["box"][3]
                    ) / 2.0

                    # vérification de la position de la box dans la ligne
                    if y_min <= box_y < y_max:

                        # récupération du nom de la colonne
                        col_name = CSV_COLUMNS[
                            b["column"] - 1
                        ]

                        # texte HTR si disponible
                        # sinon ==> chemin du crop
                        content = (
                            b["text"]
                            if b["text"]
                            else b["crop_path"]
                        )

                        # ajout du contenu dans la cellule
                        # plusieurs éléments ==> concaténés dans la même cellule
                        row_data[col_name] = (
                            row_data[col_name]
                            + " "
                            + content
                        ).strip()

                # ajout de la ligne aux résultats
                rows.append(row_data)

    # création du DataFrame avec toutes les lignes
    df = pd.DataFrame(rows)

    # chemin du CSV final
    csv_path = (
        dirs["bbox"]
        / "resultats_ocr.csv"
    )

    # export du DataFrame en CSV
    # utf-8-sig pour garder les accents correctement dans Excel
    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    return csv_path, df
