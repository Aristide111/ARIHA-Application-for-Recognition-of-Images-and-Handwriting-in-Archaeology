import pandas as pd
import ollama
from config import CLEANER_LLM_MODEL, DESTINATIONS_VALIDES, CORRECTIONS_DIRECTES


# Nettoyage de la colonne par LLM

def clean_destination(text: str) -> str:
    if not text or pd.isna(text):
        return "?"

    text_clean = str(text).strip().lower()

    if text_clean in CORRECTIONS_DIRECTES:
        return CORRECTIONS_DIRECTES[text_clean]

    prompt = (
        f"Tu es un expert en archéologie. Voici une chaîne brute extraite par HTR : '{text}'.\n"
        f"Sélectionne la destination exacte la plus proche parmi cette liste de destinations valides :\n"
        f"{', '.join(DESTINATIONS_VALIDES)}\n"
        f"Si aucune ne correspond, réponds '?'. Réponds UNIQUEMENT avec le nom exact sélectionné."
    )

    try:
        res = ollama.generate(model=CLEANER_LLM_MODEL, prompt=prompt)
        cleaned = res.get("response", "").strip()
        if cleaned in DESTINATIONS_VALIDES:
            return cleaned
        return text
    except Exception:
        return text


# Applique le traitement de nettoyage à l'ensemble du DataFrame.
def process_dataframe_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    if "Destination" in df.columns:
        df["Destination_Brute"] = df["Destination"]
        df["Destination"] = df["Destination"].apply(clean_destination)
    return df