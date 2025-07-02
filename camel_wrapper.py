
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from pathlib import Path

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from typing import Any

class LamaWrapper() :
    def __init__(self):
        # --- 1. Configuration ---

        self.PDF_FILES = []
        # Le nom de la collection que nous avons créée dans le script précédent.
        self.COLLECTION_NAME = "large_language_pizza_collection"

        

        self.load_pdf_files()

        



        # Le modèle d'embedding (doit être le même que celui utilisé pour la création).
        self.EMBEDDING_MODEL = "mxbai-embed-large"
        # Le modèle de LLM à utiliser pour la génération de la réponse.
        self.LLM_MODEL = "llama3.2:3b"

        # --- 2. Initialisation des composants LangChain ---

        print("Initialisation des composants LangChain...")

        # Initialise le client Ollama pour les embeddings
        self.ollama_embeddings = OllamaEmbeddings(model=self.EMBEDDING_MODEL)

        # Initialise le client ChromaDB pour se connecter à la base de données existante.
        # Le chemin doit correspondre à l'endroit où la DB a été créée par module5_creation_db.py
        # (qui est dans le même dossier 'code')
        self.vectorstore = Chroma(
            client=chromadb.PersistentClient(path="./chroma_db"),
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.ollama_embeddings
        )

        # Crée un retriever à partir du vectorstore.
        # Le retriever est responsable de la recherche des documents pertinents.
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3}) # Récupère les 3 chunks les plus pertinents

        # Initialise le modèle de chat Ollama
        self.llm = ChatOllama(model=self.LLM_MODEL)

        # --- 3. Définition du prompt RAG ---

        # Le template du prompt pour le LLM.
        # Il inclut le contexte récupéré et la question de l'utilisateur.
        self.template = """Réponds à la question en te basant uniquement sur le contexte suivant:
        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

        # --- 4. Construction de la chaîne RAG avec LangChain Expression Language (LCEL) ---

        # La chaîne RAG est construite en utilisant LCEL pour une meilleure lisibilité et modularité.
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} # Étape de recherche (Retrieval)
            | self.prompt                                                  # Étape d'augmentation (Augmented)
            | self.llm                                                     # Étape de génération (Generation)
            | StrOutputParser()                                            # Parse la sortie du LLM en chaîne de caractères
        )

        self.rag_chain : RunnableSerializable[Any, str] = self.rag_chain


    def load_pdf_files(self):
        pdf_directory_path = Path("PDF_Files")
        self.pdf_filepathes = []
        for file_path in list(pdf_directory_path.rglob("*.pdf")) :
            self.pdf_filepathes.append(file_path)   

                 





        def load_and_chunk_pdf(self, file_path, chunk_size=1000, chunk_overlap=200):
            """
            Charge un fichier PDF, en extrait le texte et le découpe en morceaux (chunks).
            """
            print(f"Chargement du fichier : {file_path}")
            reader = PdfReader(file_path)
            text = "".join(page.extract_text() for page in reader.pages)
            print(f"Le document contient {len(text)} caractères.")

            print("Découpage du texte en chunks...")
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                # print(text[i : i + chunk_size])
                chunks.append(text[i : i + chunk_size])
            print(f"{len(chunks)} chunks ont été créés.")
            return chunks


    