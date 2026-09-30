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


# Interface Gradio


def create_app() -> gr.Blocks:

    # création des dossiers de travail
    dirs = ensure_directories()

    # création de l'application Gradio
    with gr.Blocks(
        title="Traitement HTR",
        theme=gr.themes.Soft()
    ) as app:

        # titre
        gr.Markdown(
            "ARIHA : Application for Recognition of Images and Handwriting in Archaeology"
        )

       
        # Chargement des documents en drag and drop

        gr.Markdown("### Chargement des Documents")

        with gr.Row():

            # zone d'import
            files_upload = gr.File(
                label="Déposer les fichiers (PDF ou Images)",
                file_count="multiple",
                file_types=[
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg"
                ]
            )

            # découpage des pages
            btn_validate_upload = gr.Button(
                "Valider et découper le corpus en pages séparées",
                variant="primary"
            )

        # statut après découpae
        upload_log = gr.Textbox(
            label="Statut du découpage",
            interactive=False
        )

        gr.Markdown("---")

  
       #Calibration des colonnes

        gr.Markdown("### Calibration des colonnes du tableau")

        with gr.Row():

            with gr.Column(scale=3):

                # image utilisée pour choisir les séparations des colonnes

                calib_img = gr.Image(
                    label="Page de calibration",
                    interactive=False
                )

                with gr.Row():

                    # sélection d'une page à calibrer

                    btn_pick_page = gr.Button(
                        "Choisir une page à calibrer"
                    )

                    # réinitialisation

                    btn_reset_clicks = gr.Button(
                        "Réinitialiser les séparations"
                    )

            # affichage de l'avancement des clics, permet d econtraindre à 8 pour éviter des découpages supplémentaires ou en moins ( notamment utile quand des titres disparaissnet comme period dans certain fonds)

            calib_status = gr.Textbox(
                value="Avis : 0 / 8 clics posés",
                label="Progression",
                interactive=False
            )

            # bouton activé quand les 8 séparations sont posées

            btn_apply_calib = gr.Button(
                "Valider la calibration de la page",
                variant="primary",
                interactive=False
            )

            with gr.Column(scale=1):

                #  légende des colonnes
                gr.Markdown("#### Légende")
                gr.HTML(
                    value=generate_legend_html()
                )

        # Mosaïque pour contrôler les résultats de calibration
        # clic pour rejeter
        gr.Markdown(
            "#### Contrôle des résultats (Cliquez pour désélectionner / rejeter une page)"
        )

        gallery = gr.Gallery(
            label="Pages du lot",
            columns=3,
            height=700,
            allow_preview=False,
            object_fit="contain"
        )

        # statut global des pages du lot
        gallery_status = gr.Textbox(
            label="Statut global du lot",
            interactive=False
        )

        # Re calibration qui permet d'appliquer une sélection générale
        btn_reloop = gr.Button(
            "Re-calibrer les pages rejetées",
            variant="stop"
        )

        gr.Markdown("---")

        # Section 3 : Exécution du pipeline

        gr.Markdown("### Exécution")

        # lance YOLO + HTR + nettoyage LLM
        btn_execute = gr.Button(
            "Lancer l'HTR",
            variant="primary",
            size="lg"
        )

        # affichage du résultat du traitement
        exec_status = gr.Textbox(
            label="Journal d'exécution",
            interactive=False
        )

        gr.Markdown("---")

     
        # Section 4 : Résultats

        gr.Markdown("### Aperçu & Téléchargement")

        # CSV final à télécharger
        file_download = gr.File(
            label="Télécharger le CSV"
        )

        # aperçu du DataFrame dans l'interface
        df_preview = gr.Dataframe(
            label="Aperçu du tableau final",
            interactive=False,
            wrap=True
        )

        # Fonctions backend Gradio

        # importe les fichiers puis prépare les pages du traitement
        # découpage et miniatures
        # les pages sont ensuite placées dans pending_pages pour la calibration
        def handle_upload(files):

            # si aucun fichier ==> arrêt
            if not files:
                return "Aucun fichier importé."

            # préparation des documents importés
            GLOBAL_STATE["prepared_pages"] = save_and_split_documents(
                files,
                dirs
            )

            # toutes les pages sont en attente de calibration
            GLOBAL_STATE["pending_pages"] = list(
                GLOBAL_STATE["prepared_pages"]
            )

            # remise à zéro des ratios de calibration
            GLOBAL_STATE["page_ratios"] = {}

            return (
                f"{len(GLOBAL_STATE['prepared_pages'])} page(s) extraite(s) "
                f"et miniature(s) optimisée(s) générée(s)."
            )

        # appel de la fonction lorsque l'utilisateur valide l'import
        btn_validate_upload.click(
            fn=handle_upload,
            inputs=[files_upload],
            outputs=[upload_log]
        )

        # sélectionne une page parmi celles qui restent à calibrer
        # réinitialise les clics précédents
        # affiche la page choisie dans l'interface
        def pick_page_for_calib():

            # s'il n'y a plus de page ==> aucune calibration à faire
            if not GLOBAL_STATE["pending_pages"]:
                return (
                    None,
                    "Aucune page disponible à calibrer.",
                    gr.Button(interactive=False)
                )

            # sélection aléatoire d'une page
            sample = random.choice(
                GLOBAL_STATE["pending_pages"]
            )

            # mémorisation de la page active
            GLOBAL_STATE["current_sample"] = sample

            # remise à zéro des séparations
            GLOBAL_STATE["clicked_x"] = []

            # ouverture de l'image
            img = Image.open(
                sample[1]
            ).convert("RGB")

            return (
                img,
                f"Page active : {sample[0]} | 0 / 8 clics posés",
                gr.Button(interactive=False)
            )

        # déclenche la sélection d'une page
        btn_pick_page.click(
            fn=pick_page_for_calib,
            outputs=[
                calib_img,
                calib_status,
                btn_apply_calib
            ]
        )

        # récupère les clics pour placer les séparations
        # chaque clic correspond à une limite entre deux colonnes
        # les lignes sont redessinées après chaque clic

        def handle_click(evt: gr.SelectData):

            # aucune page sélectionnée ==> impossible de calibrer
            if not GLOBAL_STATE["current_sample"]:
                return (
                    gr.Skip(),
                    "Sélectionnez d'abord une page.",
                    gr.Button(interactive=False)
                )

            # recharge l'image originale
            sample_img = Image.open(
                GLOBAL_STATE["current_sample"][1]
            ).convert("RGB")

            # maximum 8 clics pour obtenir 9 colonnes
            if len(GLOBAL_STATE["clicked_x"]) < 8:
                GLOBAL_STATE["clicked_x"].append(
                    float(evt.index[0])
                )

            # redessine les lignes sur l'image
            updated = draw_calibration_lines(
                sample_img,
                GLOBAL_STATE["clicked_x"]
            )

            # le bouton est activé une fois les 8 clics effectués
            ready = (
                len(GLOBAL_STATE["clicked_x"]) == 8
            )

            return (
                updated,
                f"{len(GLOBAL_STATE['clicked_x'])} / 8 clics posés",
                gr.Button(interactive=ready)
            )

        # déclenche la fonction lorsqu'un clic est fait sur l'imag

        calib_img.select(
            fn=handle_click,
            outputs=[
                calib_img,
                calib_status,
                btn_apply_calib
            ]
        )

        # remet à zéro les séparations de la page actuelle
        # recharge l'image sans les lignes de calibration
        def reset_clicks():

            # aucune page active ==> rien à réinitialiser
            if not GLOBAL_STATE["current_sample"]:
                return (
                    None,
                    "Aucune page.",
                    gr.Button(interactive=False)
                )

            # suppression des clics
            GLOBAL_STATE["clicked_x"] = []

            # recharge l'image originale
            img = Image.open(
                GLOBAL_STATE["current_sample"][1]
            ).convert("RGB")

            return (
                img,
                "0 / 8 clics posés",
                gr.Button(interactive=False)
            )

        # déclenche la réinitialisation
        btn_reset_clicks.click(
            fn=reset_clicks,
            outputs=[
                calib_img,
                calib_status,
                btn_apply_calib
            ]
        )

        # transforme les 8 clics en ratios pour les 9 colonnes
        # applique ces ratios à toutes les pages du lot
        # génère les aperçus pour vérifier les résultats

        def apply_calibration():

            # récupération de la largeur de l'image
            w, _ = Image.open(
                GLOBAL_STATE["current_sample"][1]
            ).size

            # tri des clics de gauche à droite
            sorted_x = sorted(
                GLOBAL_STATE["clicked_x"]
            )

            # ajout des bords gauche et droit de l'image
            bounds = (
                [0.0]
                + sorted_x
                + [float(w)]
            )

            # calcul de la largeur relative de chaque colonne
            GLOBAL_STATE["current_batch_ratios"] = [
                (
                    bounds[i + 1]
                    - bounds[i]
                ) / w
                for i in range(len(bounds) - 1)
            ]

            # remise à zéro des pages rejetées
            GLOBAL_STATE["rejected_stems"] = set()

            # remise à zéro des aperçus en cache
            GLOBAL_STATE["cached_previews"] = {}

            items = []

            # génération d'un aperçu pour chaque page du lot
            for stem, _ in GLOBAL_STATE["pending_pages"]:

                prev_path = generate_and_cache_preview(
                    stem,
                    dirs,
                    GLOBAL_STATE["current_batch_ratios"]
                )

                # conservation du chemin de l'aperçu
                GLOBAL_STATE["cached_previews"][stem] = prev_path

                items.append(
                    (
                        str(prev_path),
                        f"OK : {stem}"
                    )
                )

            return (
                items,
                f"Lot calibré ({len(GLOBAL_STATE['pending_pages'])} pages). "
                f"Cliquez sur une image pour la rejeter."
            )

        # validation de la calibration
        btn_apply_calib.click(
            fn=apply_calibration,
            outputs=[
                gallery,
                gallery_status
            ]
        )

        # gère le clic sur une page de la galerie
        # ajoute ou retire la page de la liste des pages rejetées
        # met à jour les labels affichés dans la galerie

        def handle_gallery_select(evt: gr.SelectData):

            # récupération de la page sélectionnée
            stem = GLOBAL_STATE["pending_pages"][
                evt.index
            ][0]

            # si déjà rejetée ==> elle repasse en valide
            if stem in GLOBAL_STATE["rejected_stems"]:
                GLOBAL_STATE["rejected_stems"].remove(stem)

            # sinon ==> elle est ajoutée aux pages rejetées
            else:
                GLOBAL_STATE["rejected_stems"].add(stem)

            items = []

            # reconstruction de la galerie avec le nouveau statut
            for s, _ in GLOBAL_STATE["pending_pages"]:

                prev_path = (
                    GLOBAL_STATE["cached_previews"][s]
                )

                rej = (
                    s in GLOBAL_STATE["rejected_stems"]
                )

                label = (
                    f"REJET : {s}"
                    if rej
                    else f"OK : {s}"
                )

                items.append(
                    (
                        str(prev_path),
                        label
                    )
                )

            return (
                items,
                f"{len(GLOBAL_STATE['rejected_stems'])} "
                f"page(s) sélectionnée(s) pour rejet."
            )

        # déclenche le changement de statut lorsqu'une page est sélectionnée
        gallery.select(
            fn=handle_gallery_select,
            outputs=[
                gallery,
                gallery_status
            ]
        )

        # sépare les pages validées et les pages rejetées
        # les pages valides gardent leurs ratios
        # les pages rejetées retournent dans la file de calibration
        def reloop_rejected():

            # aucune page à traiter
            if not GLOBAL_STATE["pending_pages"]:
                return (
                    None,
                    "Aucune page.",
                    [],
                    "Terminé."
                )

            new_pending = []

            # parcours des pages du lot
            for s, path in GLOBAL_STATE["pending_pages"]:

                # page rejetée ==> retour dans la file de calibration
                if s in GLOBAL_STATE["rejected_stems"]:
                    new_pending.append(
                        (s, path)
                    )

                # page validée ==> conservation des ratios
                else:
                    GLOBAL_STATE["page_ratios"][s] = (
                        GLOBAL_STATE["current_batch_ratios"]
                    )

            # mise à jour des pages restantes
            GLOBAL_STATE["pending_pages"] = new_pending

            # remise à zéro des rejets
            GLOBAL_STATE["rejected_stems"] = set()

            # si aucune page ne reste ==> calibration terminée
            if not new_pending:
                return (
                    None,
                    "100% des pages du lot sont valides et calibrées !",
                    [],
                    "Toutes les pages sont validées !"
                )

            # sélection d'une nouvelle page parmi les pages rejetées
            sample = random.choice(
                new_pending
            )

            GLOBAL_STATE["current_sample"] = sample
            GLOBAL_STATE["clicked_x"] = []

            # affichage de la nouvelle page
            img = Image.open(
                sample[1]
            ).convert("RGB")

            return (
                img,
                f"Nouveau tour pour {len(new_pending)} page(s) rejetée(s). "
                f"Page active : {sample[0]}",
                [],
                f"{len(new_pending)} page(s) restante(s) à calibrer."
            )

        # relance la calibration sur les pages rejetées
        btn_reloop.click(
            fn=reloop_rejected,
            outputs=[
                calib_img,
                calib_status,
                gallery,
                gallery_status
            ]
        )

        # lance le pipeline complet
        # YOLO ==> HTR ==> création du CSV ==> nettoyage LLM
        # utilise les ratios enregistrés pour chaque page

        def run_full():

            return executer_pipeline_complet(
                GLOBAL_STATE["prepared_pages"],
                GLOBAL_STATE["page_ratios"],
                GLOBAL_STATE["current_batch_ratios"],
                dirs
            )

        # lancement du traitement final
        btn_execute.click(
            fn=run_full,
            outputs=[
                exec_status,
                file_download,
                df_preview
            ]
        )

    return app
