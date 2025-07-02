
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from typing import Any, List

class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, embedding_model: OllamaEmbeddings):
        self.embedding_model = embedding_model

    def __call__(self, input: List[str]):
        return self.embedding_model.embed_documents(input)

class LamaWrapper() :
    def __init__(self):
        # --- 1. Configuration ---

        # Le nom de la collection que nous avons créée dans le script précédent.
        self.COLLECTION_NAME = "large_language_pizza_collection"

        # Le modèle d'embedding (doit être le même que celui utilisé pour la création).
        self.EMBEDDING_MODEL = "mxbai-embed-large"

        # Le modèle de LLM à utiliser pour la génération de la réponse.
        self.LLM_MODEL = "llama3.2:3b"

        # --- 2. Initialisation des composants LangChain ---

        print("Initialisation des composants LangChain...")

        # Initialise le client ChromaDB
        self.persistent_client = chromadb.PersistentClient(path="./chroma_db")

        # Supprimer l'ancienne collection si elle existe
        self.clear_existing_collection()

         # Initialise le client Ollama pour les embeddings
        self.ollama_embeddings = OllamaEmbeddings(model=self.EMBEDDING_MODEL)

        # Créez une nouvelle collection avec la bonne dimension d'embedding
        self.collection = self.persistent_client.create_collection(
            name=self.COLLECTION_NAME, 
            embedding_function=CustomEmbeddingFunction(self.ollama_embeddings))
        
        # Pas obligatoire de le faire maintenant
        self.pdf_filepathes = []
        self.populate_collection()

        # Initialise le vectorstore Chroma
        self.vectorstore = Chroma(
            client=self.persistent_client,
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

        # L'assistant répond aux clients d'une pizzeria et leur permet de consulter instantanément 
        #     - les ingrédients des pizzas, 
        #     - les allergènes et informations nutritionnelles des pizzas 
        self.template = """<s>[INST] Vous êtes un assistant pour une pizzeria. Répondez à la question du client en vous basant uniquement sur le contexte suivant :
            {context}

            Répondez de manière concise et informative. Si la réponse n'est pas dans le contexte, dites simplement que vous ne savez pas.

            Question: {question}

            Instructions supplémentaires:
            - Fournissez des détails sur les ingrédients des pizzas si demandé.
            - Mentionnez les allergènes présents dans les pizzas si cela est pertinent pour la question.
            - Donnez des informations nutritionnelles si cela est demandé.
            [/INST]</s>
            [INST] {question} [/INST]"""
        
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

    def clear_existing_collection(self):
        # Vérifie si la collection existe et la supprime
        if self.COLLECTION_NAME in [collection.name for collection in self.persistent_client.list_collections()]:
            self.persistent_client.delete_collection(name=self.COLLECTION_NAME)
            print(f"Collection '{self.COLLECTION_NAME}' supprimée.")

    def populate_collection(self):
        pdf_directory_path = Path("PDF_Files")
        self.pdf_filepathes = list(pdf_directory_path.rglob("*.pdf"))
        for pdf_file_path in self.pdf_filepathes :
            # Charge et découpe le PDF
            pdf_chunks = self.load_and_chunk_pdf(pdf_file_path)
            self.collection.add(
                documents=pdf_chunks, 
                ids=[f"chunk_{i}" for i in range(len(pdf_chunks))]
            )


    def load_and_chunk_pdf(self, file_path, chunk_size=1000, chunk_overlap=200):
        """
        Charge un fichier PDF, en extrait le texte et le découpe en morceaux (chunks).
        """
        print(f"Chargement du fichier : {file_path}")
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages)
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            # print(text[i : i + chunk_size])
            chunks.append(text[i : i + chunk_size])

        print(f"{len(chunks)} chunks ont été créés.")
        return chunks


    