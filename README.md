# ARIHA : Application for Recognition of Images and Handwriting in Archaeology

ARIHA est une application de transformation de tableaux archéologiques manuels en fichiers CSV.

Elle a pour vocation à améliorer le traitement de ce type de fond en proposant une première transcription à compléter manuellement.

Elle combine un modèle de détection YOLO pour la détection de la mise en page des tableaux, un modèles d'HTR local pour la transcription, et un nettoyage intelligent par LLM pour structurer les données dans un tableau CSV exploitable.

L'entièreté de ce pipeline est local, il est donc nécessaire de l'executer via une machine équipée d'un GPU.

---

## Fonctionnement du pipeline

1. **Chargement & Découpage :** Import de fichiers PDF ou d'images (PNG, JPG) et découpage automatique en pages individuelles.

2. **Calibration des Colonnes :** Interface interactive permettant de définir les limites verticales des colonnes du tableau par simple clic sur une page de référence.

3. **Contrôle Qualité Visuel :** Mosaïque de contrôle des pages du lot pour valider ou rejeter rapidement les pages nécessitant un retraitement. Le retraitement en boucle permet de calibrer rapidement de grands volumes.

4. **Pipeline Global (YOLO + HTR + LLM) :**

   * Détection des blocs (texte, numéros, illustrations, titres) via YOLO fine-tuné sur les cahiers de fouilles de Jéricho.
   * Transcription HTR/OCR par `glm-ocr:latest`.
   * Normalisation et nettoyage intelligent des données de la colonne Destination via `qwen2.5:14b`.

5. **Aperçu & Export :** Visualisation dynamique du tableau final et téléchargement direct au format CSV.

---

## Architecture du projet

```text
├── app.py              # Point d'entrée de l'interface utilisateur (Gradio)
├── config.py           # Configuration globale, chemins et dictionnaires de référence
├── ocr_engine.py       # Moteur de traitement (découpage PDF, YOLO, HTR, gestion des aperçus)
├── pipeline.py         # Orchestration de l'exécution globale du pipeline
├── llm_cleaner.py      # Nettoyage et normalisation de la colonne Destination par LLM
└── requirements.txt    # Dépendances du projet
```

---

# Installation

## Prérequis

ARIHA nécessite :

* **Python 3.10 ou supérieur**
* **Ollama**


## Windows — Invite de commandes (`cmd`)


```cmd
cd chemin\vers\ARIHA
```

### 3. Créer un environnement virtuel

```cmd
python -m venv .venv
```

### 4. Activer l'environnement virtuel

```cmd
.venv\Scripts\activate
```

### 6. Installer les dépendances


```cmd
pip install -r requirements.txt
```

---

# Installation d'Ollama

ARIHA utilise **Ollama** pour exécuter localement les modèles de langage utilisés par le pipeline.

Après installation d'Ollama, vérifier son fonctionnement avec :

```cmd
ollama --version
```

Puis télécharger le modèle utilisé pour la transcription :

```cmd
ollama pull glm-ocr:latest
```
```cmd
ollama pull "qwen2.5:14b"
```

# Lancement de ARIHA

Une fois l'environnement virtuel activé et les dépendances installées, lancer l'application avec :

```cmd
python app.py
```

# Utilisation

Le traitement d'un corpus suit les étapes suivantes :

1. Charger un PDF ou un ensemble d'images.
2. Découper automatiquement le document en pages si nécessaire.
3. Sélectionner une page de référence.
4. Définir les limites verticales des colonnes.
5. Générer les aperçus des pages.
6. Contrôler visuellement les pages.
7. Lancer le pipeline global.
8. Effectuer la détection des blocs avec YOLO.
9. Transcrire les blocs avec `glm-ocr:latest`.
10. Nettoyer et normaliser les données avec `qwen2.5:14b`
11. Vérifier le tableau obtenu.
12. Exporter les résultats au format CSV.

---

# Modèle YOLO

Le chemin vers le modèle peut être modifié dans :

```text
config.py
```

# ARIHA : Application for Recognition of Images and Handwriting in Archaeology

ARIHA is an application designed to transform manually created archaeological tables into CSV files.

It aims to improve the processing of this type of archival material by providing an initial transcription that can then be manually completed and corrected.

It combines a YOLO detection model for identifying the layout of tables, a local HTR model for transcription, and intelligent LLM-based cleaning to structure the data into an exploitable CSV table.

The entire pipeline runs locally, so it must be executed on a machine equipped with a GPU.

---

## Pipeline Overview

1. **Loading & Splitting:** Import PDF or image files (PNG, JPG) and automatically split them into individual pages.

2. **Column Calibration:** An interactive interface allows users to define the vertical boundaries of the table columns by simply clicking on a reference page.

3. **Visual Quality Control:** A mosaic view of the pages in the batch allows users to quickly validate or reject pages requiring further processing. The iterative processing loop makes it possible to quickly calibrate large volumes of documents.

4. **Global Pipeline (YOLO + HTR + LLM):**

   * Detection of blocks (text, numbers, illustrations, titles) using a YOLO model fine-tuned on the Jericho excavation notebooks.
   * HTR/OCR transcription using `glm-ocr:latest`.
   * Intelligent normalization and cleaning of the **Destination** column using `qwen2.5:14b`.

5. **Preview & Export:** Dynamic visualization of the resulting table and direct export in CSV format.

---

## Project Architecture

```text
├── app.py              # Entry point for the user interface (Gradio)
├── config.py           # Global configuration, paths, and reference dictionaries
├── ocr_engine.py       # Processing engine (PDF splitting, YOLO, HTR, preview management)
├── pipeline.py         # Global pipeline orchestration
├── llm_cleaner.py      # LLM-based cleaning and normalization of the Destination column
└── requirements.txt    # Project dependencies
```

---

# Installation

## Prerequisites

ARIHA requires:

* **Python 3.10 or higher**
* **Ollama**
* A machine equipped with a **GPU**

---

## Windows — Command Prompt (`cmd`)

### 1. Navigate to the project directory

```cmd
cd path\to\ARIHA
```

### 2. Create a virtual environment

```cmd
python -m venv .venv
```

### 3. Activate the virtual environment

```cmd
.venv\Scripts\activate
```

### 4. Install the dependencies

```cmd
pip install -r requirements.txt
```

---

# Ollama Installation

ARIHA uses **Ollama** to run the language models locally as part of the processing pipeline.

After installing Ollama, check that it is working correctly:

```cmd
ollama --version
```

Then download the model used for transcription:

```cmd
ollama pull glm-ocr:latest
```

Download the model used for data cleaning and normalization:

```cmd
ollama pull qwen2.5:14b
```

You can check that both models are available with:

```cmd
ollama list
```

The following models should appear in the list:

```text
glm-ocr:latest
qwen2.5:14b
```

---

# Running ARIHA

Once the virtual environment is activated and all dependencies have been installed, launch the application with:

```cmd
python app.py
```

The Gradio interface will then be available locally.

---

# Usage

The processing of an archaeological corpus follows these steps:

1. Load a PDF or a set of images.
2. Automatically split the document into individual pages if necessary.
3. Select a reference page.
4. Define the vertical boundaries of the columns.
5. Generate page previews.
6. Perform visual quality control.
7. Launch the global pipeline.
8. Detect table blocks using YOLO.
9. Transcribe the blocks using `glm-ocr:latest`.
10. Clean and normalize the data using `qwen2.5:14b`.
11. Review and manually correct the resulting table.
12. Export the results in CSV format.

---

# YOLO Model

ARIHA uses a YOLO model specifically trained to detect the different elements present in the archaeological tables.

The path to the YOLO model can be configured in:

```text
config.py
```
