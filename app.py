import random
from pathlib import Path
import gradio as gr
import pandas as pd
from PIL import Image

from config import GLOBAL_STATE
from ocr_engine import (
    ensure_directories,
    save_and_split_documents,
    generate_legend_html,
    draw_calibration_lines,
    generate_and_cache_preview
)
from pipeline import executer_pipeline_complet


# Crée et configure l'interface utilisateur Gradio pour le pipeline de traitement HTR. ( frontend)

def create_app() -> gr.Blocks:
    dirs = ensure_directories()

    with gr.Blocks(title="Traitement HTR", theme=gr.themes.Soft()) as app:
        gr.Markdown("ARIHA : Application for Recognition of Images and Handwriting in Archaeology")

        # découpage des documents
        
        gr.Markdown("### Chargement des Documents")
        with gr.Row():
            files_upload = gr.File(
                label="Déposer les fichiers (PDF ou Images)",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"]
            )
            btn_validate_upload = gr.Button("Valider & découper le corpus en pages séparées", variant="primary")
        
        upload_log = gr.Textbox(label="Statut du découpage", interactive=False)

        gr.Markdown("---")

        # calibration manuelle des colonnes

        gr.Markdown("### Calibration des colonnes du tableau")
        with gr.Row():
            with gr.Column(scale=3):
                calib_img = gr.Image(label="Page de calibration", interactive=False)
                
                with gr.Row():
                    btn_pick_page = gr.Button("Choisir une page à callibrer")
                    btn_reset_clicks = gr.Button("Réinitialiser les séparations")

                calib_status = gr.Textbox(value="Avis : 0 / 8 clics posés", label="Progression", interactive=False)
                btn_apply_calib = gr.Button("Valider la calibration de la page", variant="primary", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("#### Légende")
                gr.HTML(value=generate_legend_html())

        gr.Markdown("#### Contrôle des résultats (Cliquez pour désélectionner / rejeter une page)")
        
        gallery = gr.Gallery(
            label="Pages du lot", 
            columns=3, 
            height=700, 
            allow_preview=False,
            object_fit="contain"
        )
        gallery_status = gr.Textbox(label="Statut global du lot", interactive=False)
        btn_reloop = gr.Button("Re-calibrer les pages rejetées", variant="stop")

        gr.Markdown("---")

        # Segmentation, HTR et nettoyage des résultats par LLM

        gr.Markdown("### Exécution")
        btn_execute = gr.Button(" Lancer l'HTR", variant="primary", size="lg")
        exec_status = gr.Textbox(label="Journal d'exécution", interactive=False)

        gr.Markdown("---")

        # Apperçu des résultats et export

        gr.Markdown("### Aperçu & Téléchargement")
        file_download = gr.File(label="Télécharger le CSV")
        df_preview = gr.Dataframe(
            label="Aperçu du tableau final",
            interactive=False,
            wrap=True
        )

        # Fonction appelées par l'application gradio (backend)

        # Découpage des pages et optimisation des miniatures pour alléger le traitement de gros volumes

        def handle_upload(files):
            if not files:
                return "Aucun fichier importé."
            GLOBAL_STATE["prepared_pages"] = save_and_split_documents(files, dirs)
            GLOBAL_STATE["pending_pages"] = list(GLOBAL_STATE["prepared_pages"])
            GLOBAL_STATE["page_ratios"] = {}
            return f" {len(GLOBAL_STATE['prepared_pages'])} page(s) extraite(s) et miniature(s) optimisée(s) générée(s)."

        btn_validate_upload.click(fn=handle_upload, inputs=[files_upload], outputs=[upload_log])

        # Sélectionne aléatoirement une page parmi les documents en attente pour lancer sa calibration.

        def pick_page_for_calib():
            if not GLOBAL_STATE["pending_pages"]:
                return None, " Aucune page disponible à calibrer.", gr.Button(interactive=False)
            sample = random.choice(GLOBAL_STATE["pending_pages"])
            GLOBAL_STATE["current_sample"] = sample
            GLOBAL_STATE["clicked_x"] = []
            img = Image.open(sample[1]).convert("RGB")
            return img, f"Page active : {sample[0]} | 0 / 8 clics posés", gr.Button(interactive=False)

        btn_pick_page.click(fn=pick_page_for_calib, outputs=[calib_img, calib_status, btn_apply_calib])

        # Enregistre les coordonnées des clics utilisateur sur l'image de calibration pour délimiter les colonnes.

        def handle_click(evt: gr.SelectData):
            if not GLOBAL_STATE["current_sample"]:
                return gr.Skip(), "Sélectionnez d'abord une page.", gr.Button(interactive=False)
            sample_img = Image.open(GLOBAL_STATE["current_sample"][1]).convert("RGB")
            if len(GLOBAL_STATE["clicked_x"]) < 8:
                GLOBAL_STATE["clicked_x"].append(float(evt.index[0]))
            updated = draw_calibration_lines(sample_img, GLOBAL_STATE["clicked_x"])
            ready = len(GLOBAL_STATE["clicked_x"]) == 8
            return updated, f"{len(GLOBAL_STATE['clicked_x'])} / 8 clics posés", gr.Button(interactive=ready)

        calib_img.select(fn=handle_click, outputs=[calib_img, calib_status, btn_apply_calib])

        # Réinitialise les clics.

        def reset_clicks():
            if not GLOBAL_STATE["current_sample"]:
                return None, "Aucune page.", gr.Button(interactive=False)
            GLOBAL_STATE["clicked_x"] = []
            img = Image.open(GLOBAL_STATE["current_sample"][1]).convert("RGB")
            return img, "0 / 8 clics posés", gr.Button(interactive=False)

        btn_reset_clicks.click(fn=reset_clicks, outputs=[calib_img, calib_status, btn_apply_calib])

        # Calcule les ratios de colonnes et génère les aperçus pour toutes les pages.

        def apply_calibration():
            w, _ = Image.open(GLOBAL_STATE["current_sample"][1]).size
            sorted_x = sorted(GLOBAL_STATE["clicked_x"])
            bounds = [0.0] + sorted_x + [float(w)]
            GLOBAL_STATE["current_batch_ratios"] = [(bounds[i+1] - bounds[i]) / w for i in range(len(bounds) - 1)]
            GLOBAL_STATE["rejected_stems"] = set()
            GLOBAL_STATE["cached_previews"] = {}
            
            items = []
            for stem, _ in GLOBAL_STATE["pending_pages"]:
                prev_path = generate_and_cache_preview(stem, dirs, GLOBAL_STATE["current_batch_ratios"])
                GLOBAL_STATE["cached_previews"][stem] = prev_path
                items.append((str(prev_path), f"OK : {stem}"))

            return items, f"Lot calibré ({len(GLOBAL_STATE['pending_pages'])} pages). Cliquez sur une image pour la rejeter."

        btn_apply_calib.click(fn=apply_calibration, outputs=[gallery, gallery_status])

        # Gère la sélection ou la désélection d'une page dans la galerie pour marquer un rejet ou une validation.

        def handle_gallery_select(evt: gr.SelectData):
            stem = GLOBAL_STATE["pending_pages"][evt.index][0]
            
            if stem in GLOBAL_STATE["rejected_stems"]:
                GLOBAL_STATE["rejected_stems"].remove(stem)
            else:
                GLOBAL_STATE["rejected_stems"].add(stem)

            items = []
            for s, _ in GLOBAL_STATE["pending_pages"]:
                prev_path = GLOBAL_STATE["cached_previews"][s]
                rej = s in GLOBAL_STATE["rejected_stems"]
                label = f"REJET : {s}" if rej else f"OK : {s}"
                items.append((str(prev_path), label))
                
            return items, f"{len(GLOBAL_STATE['rejected_stems'])} page(s) sélectionnée(s) pour rejet."

        gallery.select(fn=handle_gallery_select, outputs=[gallery, gallery_status])

        # Relance le processus de calibration pour les pages ayant été rejetées par l'utilisateur.
        # La boucle permet de gérer des gros volumes rapidement

        def reloop_rejected():
            if not GLOBAL_STATE["pending_pages"]:
                return None, "Aucune page.", [], "Terminé."

            new_pending = []
            for s, path in GLOBAL_STATE["pending_pages"]:
                if s in GLOBAL_STATE["rejected_stems"]:
                    new_pending.append((s, path))
                else:
                    GLOBAL_STATE["page_ratios"][s] = GLOBAL_STATE["current_batch_ratios"]

            GLOBAL_STATE["pending_pages"] = new_pending
            GLOBAL_STATE["rejected_stems"] = set()

            if not new_pending:
                return None, " 100% des pages du lot sont valides et calibrées !", [], "Toutes les pages sont validées !"

            sample = random.choice(new_pending)
            GLOBAL_STATE["current_sample"] = sample
            GLOBAL_STATE["clicked_x"] = []
            img = Image.open(sample[1]).convert("RGB")
            return img, f"Nouveau tour pour {len(new_pending)} page(s) rejetée(s). Page active : {sample[0]}", [], f"{len(new_pending)} page(s) restante(s) à calibrer."

        btn_reloop.click(fn=reloop_rejected, outputs=[calib_img, calib_status, gallery, gallery_status])

        # Exécute l'ensemble du pipeline complet de traitement OCR, d'extraction et de nettoyage des données.
        
        def run_full():
            return executer_pipeline_complet(
                GLOBAL_STATE["prepared_pages"],
                GLOBAL_STATE["page_ratios"],
                GLOBAL_STATE["current_batch_ratios"],
                dirs
            )

        btn_execute.click(fn=run_full, outputs=[exec_status, file_download, df_preview])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)