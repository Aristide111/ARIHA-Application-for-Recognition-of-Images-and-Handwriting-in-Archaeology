import random
from pathlib import Path
import albumentations as A

# Configuration globale du pipeline

# Dossier d'entrée ou de sortie

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
MODEL_PATH = SCRIPT_DIR / "last.pt"

# Configuration du seuil de confiance

DPI = 200
CONF_THRESHOLD = 0.25

# Modèles HTR et LLM.

OCR_MODEL = "glm-ocr:latest"
CLEANER_LLM_MODEL = "qwen2.5:14b"

# Labels détectés par le modèle YOLO

LABELS = {0: "texte", 1: "illustration", 2: "numero", 3: "titre"}

# colomnes du CSV et couleurs

CSV_COLUMNS = [
    "Numéro", "Objet", "Description", "Illustration",
    "Numero Photo", "Numero Dessin", "Localisation", "Periode", "Destination"
]

COLUMN_COLORS_HEX = [
    "#E53935", "#43A047", "#1E88E5", "#FB8C00", "#8E24AA", 
    "#00ACC1", "#D81B60", "#7CB342", "#C2185B"
]

COLUMN_COLORS_RGBA = [
    (229, 57, 53, 70), (67, 160, 71, 70), (30, 136, 229, 70), 
    (251, 140, 0, 70), (142, 36, 170, 70), (0, 172, 193, 70), 
    (216, 27, 96, 70), (124, 179, 66, 70), (194, 24, 91, 70)
]

# Prompts pour le modèle d'OCR et définition des labels d'HTR 

HTR_LABELS = {"numero", "texte"}

OCR_PROMPT = (
    "Transcris fidèlement tout le texte visible sur cette image. "
    "Réponds uniquement avec le texte reconnu, sans commentaire ni mise en forme."
)

# définition de la transformation

INVERT_TRANSFORM = A.InvertImg(p=1.0)

# Mémoire centrale de l'application via GLOBAL STATE

GLOBAL_STATE = {
    "prepared_pages": [],        # list of (stem, Path)
    "pending_pages": [],         # pages restantes à calibrer
    "current_sample": None,      # (stem, Path)
    "clicked_x": [],             # x coordinates
    "current_batch_ratios": [],
    "rejected_stems": set(),
    "page_ratios": {},           # {stem: ratios}
    "all_bboxes": {},
}

# Liste de référence établie manuellement

DESTINATIONS_VALIDES = [
    "Amman", "Palestine Arch. Museum (Amman)", "Ashmolean", "A.S.O.R/ A.S.OR/ASOR",
    "Berlin", "Birmingham", "Birmingham (City Museum and Art Gallery)", "British Museum",
    "BM", "Cambridge", "Dublin", "Trinity College", "Durham", "Ecole biblique",
    "Ecole biblique (Jerusalem)", "Edinburgh", "Emory", "Emory U.", "Glasgow",
    "Institute of Archaeology", "I of A", "I of A (London)", "Inst. Of Archaeology (London)",
    "Inst. of Arch.", "Institute of Archaeology Australia", "IoA Australia",
    "Aust. Institute", "Australian Institute", "Aust. Inst.", "Aust.Inst. of A.",
    "Aust.Inst. of Arch.", "Jericho", "Local museum", "Local Mus.", "H. palace",
    "Hisham Palace", "Jerusalem", "Jerusalem (PAM)", "P.A.M. (Jerusalem)",
    "Palestine Arch. Museum", "Palestine Archaeol. Museum", "Palest. Arch. Museum",
    "Leeds", "Leeds University (Semitic Dept.)", "Leeds U.", "Leeds Univ.",
    "Leeds City Museum", "Leeds City", "Leiden", "Leiden University", "Liverpool",
    "Lund", "Manchester", "New York", "New York (Am. Mus. Nat. Hist.)",
    "Am. Mus. of Nat. Hist.", "American Museum of Natural History",
    "Pal. Nat. Museum (Amman)", "Palestine Nat. Museum (Amman)", "Pontifical Institute",
    "Pontif Inst.", "Pontif. Institute", "Pont. Inst.", "Pontifical Biblical Inst.",
    "St Andrews", "Stockholm", "Medelhavsmuseet", "Sydney", "Texas Christian U.",
    "Texas Christian U. (W.L. Reed 1983)", "Toronto", "Toronto (Royal Ontario Museum)",
    "Discarded", "Discard", "Disc", "Dis", "Left on site", "on site", "Disintegrated",
    "Missing", "?", "? Location", "Gen. Dist.", "For study", "For report/Sent for report",
    "On loan I of A to be distributed", "On loan for study", "Sent for identification",
    "Divided", "Amman report first", "Stone sent for analysis", "Amman (half) Cambridge"
]

# Correction pour chaque élément pour les rattacher au plus proche

CORRECTIONS_DIRECTES = {
    "d.s.o.r": "A.S.O.R/ A.S.OR/ASOR",
    "a.s.o.r.": "A.S.O.R/ A.S.OR/ASOR",
    "asor": "A.S.O.R/ A.S.OR/ASOR",
    "b.m.": "BM",
    "p.a.m.": "Palestine Arch. Museum",
    "destination": "?",
}

# Liste des symbole Ibid

SYMBOLES_DITO = {'"', '""', "''", "n", "m", "h", "id", "idem", "-", "—", ".", "do"}
#MOTS_PARASITES = {"brown", "system", "systems", "u", "inkun"}