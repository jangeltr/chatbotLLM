import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class Query(BaseModel):
    prompt: str

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ==================== CONFIGURACIÓN ====================
DB_FAISS_PATH = 'vectorstore/db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# Inicializar embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Cargar base de datos vectorial
db = FAISS.load_local(
    DB_FAISS_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# ==================== RETRIEVER Y RERANKER ====================
# MEJORA 1: Aumentar k para recuperar más documentos antes del reranking
base_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        'k': 20,  # Aumentado de 10 a 20
        'fetch_k': 50  # Buscar más candidatos antes de filtrar
    }
)

# MEJORA 2: Configurar el reranker para mantener más documentos relevantes
compressor = FlashrankRerank(
    model="ms-marco-MultiBERT-L-12",
    top_n=8  # Mantener los 8 documentos más relevantes después del reranking
)

# LLM
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="not-needed", 
    temperature=0.2,  # Aumentado ligeramente para respuestas más naturales
    callbacks=[StreamingStdOutCallbackHandler()]
)

# ==================== PROMPTS MEJORADOS ====================

# MEJORA 3: Prompt de reescritura más específico
rewrite_template = """### INSTRUCCIÓN ###
Eres un experto en búsqueda de información. Tu tarea es reformular la pregunta del usuario para encontrar información específica en documentos del TecNM campus Tlajomulco.

**REGLAS IMPORTANTES:**
1. Si preguntan por una persona (director, jefe, coordinador, etc.), reformula como: "nombre [CARGO] [ÁREA]"
2. Si preguntan por un proceso o trámite, incluye palabras clave relacionadas
3. Mantén la pregunta concisa y enfocada
4. Expande sinónimos y términos relacionados

### PREGUNTA ORIGINAL ###
{question}

### PREGUNTA OPTIMIZADA PARA BÚSQUEDA ###"""

rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
query_rewriter = rewrite_prompt | llm | StrOutputParser()

# MEJORA 4: Prompt RAG mejorado con mejor estructura
rag_template = """### ROL ###
Eres el asistente virtual oficial del TecNM campus Tlajomulco. Tu función es proporcionar información precisa y útil basada ÚNICAMENTE en los documentos oficiales de la institución.

### CONTEXTO RECUPERADO ###
{context}

### PREGUNTA DEL USUARIO ###
{question}

### INSTRUCCIONES DE RESPUESTA ###
1. **Analiza cuidadosamente** todo el contexto antes de responder
2. **Busca información directa:** Si preguntan por un cargo (director, jefe, coordinador), busca el nombre de la persona que ocupa ese puesto específico
3. **Considera sinónimos:** "directora" = "director", "jefa" = "jefe", "encargado" = "coordinador", etc.
4. **Sé específico:** Si encuentras la información, cita el cargo exacto y el nombre completo
5. **Estructura jerárquica:** Directora > Subdirectores > Coordinadores > Jefes de departamento
6. **Si NO encuentras información:** Di claramente "No encontré esa información específica en los documentos disponibles"

### FORMATO DE RESPUESTA ###
- Sé conciso y directo
- Usa bullet points si hay múltiples datos
- Cita el cargo oficial exacto
- Si hay información complementaria relevante, inclúyela

### RESPUESTA ###"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

# ==================== FUNCIONES AUXILIARES ====================

def format_docs(docs):
    """Formatea documentos con información de fuente"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"--- Documento {i} (Fuente: {source}, Página: {page}) ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted)

def retrieve_and_rerank(input_dict):
    """
    MEJORA 5: Pipeline de recuperación mejorado con logging detallado
    """
    question = input_dict["question"]
    
    # Paso 1: Reescribir la pregunta
    rewritten_question = query_rewriter.invoke({"question": question})
    print(f"\n{'='*60}")
    print(f"[PREGUNTA ORIGINAL]: {question}")
    print(f"[PREGUNTA REESCRITA]: {rewritten_question}")
    
    # Paso 2: Búsqueda inicial amplia
    documents = base_retriever.invoke(rewritten_question)
    print(f"[DOCUMENTOS INICIALES]: {len(documents)} documentos recuperados")
    
    # Paso 3: Reranking
    reranked_docs = compressor.compress_documents(documents, rewritten_question)
    print(f"[DESPUÉS DE RERANKING]: {len(reranked_docs)} documentos relevantes")
    
    # Debug: Mostrar snippet de los top 3 documentos
    print("\n[TOP 3 DOCUMENTOS MÁS RELEVANTES]:")
    for i, doc in enumerate(reranked_docs[:3], 1):
        snippet = doc.page_content[:150].replace('\n', ' ')
        print(f"  {i}. {snippet}...")
    print('='*60 + '\n')
    
    return {
        "context": format_docs(reranked_docs),
        "question": question,
        "source_documents": reranked_docs
    }

# ==================== CADENA RAG ====================

rag_chain = (
    RunnablePassthrough.assign(rag_input=retrieve_and_rerank)
    | (lambda x: {
        "context": x["rag_input"]["context"],
        "question": x["rag_input"]["question"]
    })
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

# ==================== ENDPOINTS ====================

GREETINGS = [
    "hola", "buenos días", "buenas tardes", "buenas noches", 
    "qué tal", "hey", "saludos", "buen día"
]

@app.post("/chat")
async def chatear(query: Query):
    user_prompt_cleaned = query.prompt.lower().strip()
    
    # Manejo de saludos
    if any(greeting in user_prompt_cleaned for greeting in GREETINGS) and len(user_prompt_cleaned) < 20:
        return {
            "respuesta": "¡Hola! Soy el asistente virtual del TecNM campus Tlajomulco. Puedo ayudarte con información sobre:\n- Normatividad institucional\n- Actividades escolares\n- Eventos\n- Contactos y directorio\n- Trámites académicos\n\n¿Qué necesitas saber?",
            "fuentes": []
        }
    
    try:
        # Ejecutar el pipeline RAG completo
        rag_output = retrieve_and_rerank({"question": query.prompt})
        respuesta = (rag_prompt | llm | StrOutputParser()).invoke(rag_output)
        
        # Extraer fuentes únicas
        fuentes = list(set(
            os.path.basename(doc.metadata.get("source", "N/A")) 
            for doc in rag_output["source_documents"]
        ))
        
        print(f"\n[RESPUESTA GENERADA]: {respuesta[:200]}...")
        print(f"[FUENTES]: {fuentes}\n")
        
        # MEJORA 6: Validación de respuesta mejorada
        respuesta_lower = respuesta.lower()
        palabras_inseguridad = [
            "no tengo esa información",
            "no encontré información",
            "no está disponible",
            "no puedo proporcionar"
        ]
        
        if any(palabra in respuesta_lower for palabra in palabras_inseguridad):
            fuentes = []
        
        # Si la respuesta es muy corta, es probable que no haya información
        if len(respuesta.strip()) < 20:
            respuesta = "No encontré información específica sobre tu consulta en los documentos disponibles. Te recomiendo contactar directamente a Servicios Escolares o al departamento correspondiente."
            fuentes = []
        
        return {
            "respuesta": respuesta,
            "fuentes": fuentes
        }
    
    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return {
            "respuesta": "Lo siento, hubo un error al procesar tu consulta. Por favor, intenta reformular tu pregunta.",
            "fuentes": []
        }

@app.get("/")
async def root():
    return {
        "mensaje": "API del Chatbot TecNM Tlajomulco",
        "version": "2.0",
        "endpoints": {
            "/chat": "POST - Envía una pregunta al chatbot",
            "/": "GET - Información de la API"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)