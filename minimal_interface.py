import gradio as gr

from langchain_core.runnables import RunnableSerializable
from typing import Any

from camel_wrapper import LamaWrapper

lama = LamaWrapper()

def respond(message, history):
    runnable_chain = lama.rag_chain
    try :
        lama_answer = runnable_chain.invoke(message)
        return lama_answer 
    except Exception as ex :
        return str(ex)

# Création de l'interface Gradio
minim_inter_gradio = gr.ChatInterface(
    fn=respond,
    title="Chatbot Minimal",
    description="Un exemple simple de chatbot avec Gradio",
    chatbot=gr.Chatbot(type="messages")  
)

# Lancement de l'interface
minim_inter_gradio.launch()