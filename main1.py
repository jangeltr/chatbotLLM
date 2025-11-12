import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================================================================
# === IMPORTS FINALIZADOS Y MODERNIZADOS ===
# ==============================================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI # Usaremos ChatOpenAI para mayor compatibilidad
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# ==============================================================================

class Query(BaseModel):
    prompt: str

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuración de LangChain ---

DB_FAISS_PATH = 'vectorstore/db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 5}) # Definimos el retriever aquí

# Usaremos ChatOpenAI que es más estándar para agentes e integraciones modernas
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.2,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Definimos una plantilla de prompt más compatible con modelos de chat
prompt_template = """
### INSTRUCCIÓN ###
Eres un asistente virtual experto del "TecNM campus Tlajomulco". Tu única fuente de verdad es el CONTEXTO proporcionado, que contiene extractos de los reglamentos y documentos oficiales de esta institución.

Tu tarea principal es interpretar las preguntas de los usuarios asumiendo que siempre se refieren al "TecNM campus Tlajomulco", a menos que especifiquen lo contrario.

- Si un usuario pregunta "¿Quién es la directora?", debes buscar en el CONTEXTO la información sobre la directora del "TecNM campus Tlajomulco".
- Si el CONTEXTO que se te proporciona contiene la información relevante para responder la PREGUNTA, responde de manera clara y concisa basándote SÓLO en ese contexto.
- Si el CONTEXTO no contiene la información necesaria para responder la PREGUNTA, responde amablemente que no tienes esa información específica. No intentes adivinar ni inventar respuestas.

### CONTEXTO ###
{context}

### PREGUNTA ###
{question}

### RESPUESTA ###
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# ==============================================================================
# === NUEVA FORMA DE CREAR LA CADENA (LangChain Expression Language - LCEL) ===
# ==============================================================================
# Esta es la forma moderna, explícita y recomendada de construir cadenas en LangChain.

def format_docs(docs):
    # Une el contenido de los documentos recuperados en un solo bloque de texto.
    return "\n\n".join(doc.page_content for doc in docs)

# Definimos la cadena de ejecución paso a paso:
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# ==============================================================================

GREETINGS = ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey"]

@app.post("/chat")
async def chatear(query: Query):
    user_prompt_cleaned = query.prompt.lower().strip()

    if user_prompt_cleaned in GREETINGS:
        print("Detectado saludo. Respondiendo de forma predefinida.")
        return {"respuesta": "¡Hola! Soy el asistente virtual de la institución. ¿Cómo puedo ayudarte con tus dudas sobre trámites o reglamentos?", "fuentes": []}

    print(f"Recibida pregunta: {query.prompt}")
    
    # Invocamos la nueva cadena. El input es directamente la pregunta del usuario.
    respuesta = rag_chain.invoke(query.prompt)
    
    # Para obtener las fuentes, hacemos una llamada separada al retriever
    source_documents = retriever.invoke(query.prompt)
    fuentes = list(set(os.path.basename(doc.metadata.get("source", "N/A")) for doc in source_documents))
    
    print(f"Respuesta generada: {respuesta}")
    print(f"Fuentes encontradas: {fuentes}")

    if not respuesta or len(respuesta) < 15:
        respuesta = "En este momento no tengo esa información, te recomiendo contactar a Servicios Escolares al correo servicios.escolares@tlajomco.tecn.mx"
        fuentes = []
    # Si la respuesta es la de fallback, no mostramos fuentes aleatorias
    elif "no tengo esa información" in respuesta.lower():
        fuentes = []

    return {"respuesta": respuesta, "fuentes": fuentes}