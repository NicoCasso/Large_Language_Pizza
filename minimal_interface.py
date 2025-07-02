import gradio as gr

def respond(message, history):
    # Ici, vous pouvez ajouter la logique de votre chatbot
    return "Écho: " + message

# Création de l'interface Gradio
minim_inter_gradio = gr.ChatInterface(
    fn=respond,
    inputs=gr.Textbox(label="Your Name"),
    outputs="text",
    title="Chatbot Minimal",
    description="Un exemple simple de chatbot avec Gradio"
)

# Lancement de l'interface
minim_inter_gradio.launch()