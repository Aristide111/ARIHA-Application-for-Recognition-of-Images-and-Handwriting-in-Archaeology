import pandas as pd
from pathlib import Path
from ocr_engine import run_yolo_and_htr, build_csv_output
from llm_cleaner import process_dataframe_cleaning


# Exécute l'intégralité du traitement.

def executer_pipeline_complet(
    prepared_pages: list[tuple[str, Path]],
    page_ratios: dict[str, list[float]],
    default_ratios: list[float],
    dirs: dict[str, Path]
) -> tuple[str, str, pd.DataFrame]:
    if not prepared_pages:
        return "Aucune page à traiter.", None, pd.DataFrame()

    # 1. Détection YOLO & HTR/OCR

    all_bboxes = run_yolo_and_htr(prepared_pages, page_ratios, default_ratios, dirs)

    # 2. Construction du CSV brut

    csv_path, df = build_csv_output(all_bboxes, dirs)

    # 3. Nettoyage LLM de la colonne Destination

    df_cleaned = process_dataframe_cleaning(df)
    df_cleaned.to_csv(csv_path, index=False, encoding="utf-8-sig")

    log_msg = f"Traitement effectué : {len(prepared_pages)} page(s) traitée(s).\nFichier enregistré sous : {csv_path}"

    return log_msg, str(csv_path), df_cleaned