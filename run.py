# Execute l'application gradio

from app import create_app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)