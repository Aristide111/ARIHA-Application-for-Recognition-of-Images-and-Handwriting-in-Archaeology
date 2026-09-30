
import pandas as pd
from pathlib import Path
from ocr_engine import run_yolo_and_htr, build_csv_output
from llm_cleaner import process_dataframe_cleaning

# exécute toutes les étapes du traitement dans le bon ordre

# vérifie existence des pages

# YOLO et HTR

#  CSV avec les résultats détecté

#  nettoyage LLM

# applique modification CSV

#  message de traitement, le chemin du CSV et le DataFrame final

def executer_pipeline_complet(
    prepared_pages: list[tuple[str, Path]],
    page_ratios: dict[str, list[float]],
    default_ratios: list[float],
    dirs: dict[str, Path]
) -> tuple[str, str, pd.DataFrame]:

    # si aucune page ==> arrêt du traitement
    if not prepared_pages:
        return (
            "Aucune page à traiter.",
            None,
            pd.DataFrame()
        )

    # 1. Détection YOLO et HTR

    # détecte les éléments présents sur les pages
    # lance l'HTR sur les éléments concernés
    all_bboxes = run_yolo_and_htr(
        prepared_pages,
        page_ratios,
        default_ratios,
        dirs
    )

    # 2. Construction du CSV brut
    # transforme les détections en lignes et colonnes
    csv_path, df = build_csv_output(
        all_bboxes,
        dirs
    )

    # 3. Nettoyage LLM de la colonne Destination
    # passe le DataFrame dans le LLM pour nettoyer les résultats
    df_cleaned = process_dataframe_cleaning(df)

    # sauvegarde du DataFrame nettoyé dans le même CSV
    df_cleaned.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # création du message de fin avec le nombre de pages traitées
    # et le chemin du fichier généré
    log_msg = (
        f"Traitement effectué : {len(prepared_pages)} page(s) traitée(s).\n"
        f"Fichier enregistré sous : {csv_path}"
    )

    return (
        log_msg,
        str(csv_path),
        df_cleaned
    )
