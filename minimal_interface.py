import gradio as gr

from camel_wrapper import LamaWrapper

lama = LamaWrapper()

def respond(message, history):
    lama_answer = lama.rag_chain.invoke(message)
    return lama_answer 

# Création de l'interface Gradio
minim_inter_gradio = gr.ChatInterface(
    fn=respond,
    title="Chatbot Minimal",
    description="Un exemple simple de chatbot avec Gradio",
    chatbot=gr.Chatbot(type="messages")  
)

# Lancement de l'interface
minim_inter_gradio.launch()