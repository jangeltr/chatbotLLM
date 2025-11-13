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

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

db = FAISS.load_local(
    DB_FAISS_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# ==================== RETRIEVER Y RERANKER ====================
# 🔥 MEJORA: Recuperar más documentos para análisis
base_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        'k': 30,  # Aumentado para tener más opciones
        'fetch_k': 100  # Pool más grande de candidatos
    }
)

# 🔥 MEJORA: Mantener más documentos después del reranking
compressor = FlashrankRerank(
    model="ms-marco-MultiBERT-L-12",
    top_n=8  # Aumentado a 8 para tener mejor contexto
)

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="not-needed", 
    temperature=0.1,  # Más determinista
    callbacks=[StreamingStdOutCallbackHandler()]
)

# ==================== PROMPTS MEJORADOS ====================

# 🔥 MEJORA CRÍTICA: Prompt RAG ultra-específico
rag_template = """Eres el asistente oficial del TecNM Tlajomulco. Responde ÚNICAMENTE con información del CONTEXTO proporcionado.

📋 CONTEXTO:
{context}

❓ PREGUNTA: {question}

🎯 INSTRUCCIONES CRÍTICAS:
1. Si preguntan por un CARGO (director, jefe, coordinador):
   - Busca EXACTAMENTE ese cargo en el contexto
   - Ignora cargos similares (director ≠ subdirector, jefe ≠ coordinador)
   - Formato: "[NOMBRE COMPLETO] es [CARGO EXACTO]"

2. DISTINGUIR CARGOS:
   - Director(a) ≠ Subdirector(a)
   - Jefe(a) ≠ Coordinador(a) ≠ Encargado(a)
   - Lee cuidadosamente el cargo exacto antes de responder

3. Si el contexto tiene el cargo exacto → Da el nombre
4. Si NO está el cargo exacto → "No encontré información sobre [CARGO] en los documentos"
5. NUNCA inventes información
6. NUNCA des un cargo diferente al solicitado

RESPUESTA (directa y específica):"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

# ==================== FUNCIONES AUXILIARES ====================

def format_docs(docs):
    """Formato mejorado con numeración clara y metadata"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "Desconocido"))
        # Agregar metadatos para mejor contexto
        formatted.append(f"[Documento {i} - Fuente: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def prepare_search_query(question: str) -> str:
    """Garantiza que las búsquedas incluyan el contexto del campus."""
    cleaned = question.strip()
    anchor_terms = [
        "tecnm tlajomulco",
        "tecnm campus tlajomulco",
        "instituto tecnológico de tlajomulco",
        "instituto tecnologico de tlajomulco",
        "ittj"
    ]
    lower_cleaned = cleaned.lower()
    if not any(term in lower_cleaned for term in anchor_terms):
        cleaned = f"{cleaned} TecNM Campus Tlajomulco"
    return cleaned

def retrieve_and_rerank(input_dict):
    """Pipeline de recuperación con logging detallado"""
    question = input_dict["question"]
    search_query = prepare_search_query(question)
    
    print(f"\n{'='*70}")
    print(f"📝 PREGUNTA ORIGINAL: {question}")
    print(f"🔍 CONSULTA DE BÚSQUEDA: {search_query}")
    
    # Paso 1: Búsqueda inicial
    documents = base_retriever.invoke(search_query)
    print(f"📚 DOCUMENTOS RECUPERADOS: {len(documents)}")
    
    # Paso 2: Reranking
    reranked_docs = compressor.compress_documents(documents, search_query)
    print(f"⭐ DESPUÉS DE RERANKING: {len(reranked_docs)} documentos relevantes")
    
    # 🔥 MEJORA: Mostrar más contexto en debug
    print("\n🔝 TOP 5 DOCUMENTOS MÁS RELEVANTES:")
    for i, doc in enumerate(reranked_docs[:5], 1):
        snippet = doc.page_content[:200].replace('\n', ' ')
        source = os.path.basename(doc.metadata.get("source", "N/A"))
        print(f"  {i}. [{source}] {snippet}...")
    print('='*70 + '\n')
    
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
    "qué tal", "hey", "saludos", "buen día", "buenas"
]

@app.post("/chat")
async def chatear(query: Query):
    user_prompt_cleaned = query.prompt.lower().strip()
    
    # Manejo de saludos
    if any(greeting in user_prompt_cleaned for greeting in GREETINGS) and len(user_prompt_cleaned) < 25:
        return {
            "respuesta": "¡Hola! 👋 Soy el asistente virtual del TecNM Tlajomulco.\n\nPuedo ayudarte con:\n✅ Directorio y contactos\n✅ Información de carreras\n✅ Trámites académicos\n✅ Servicio social y residencias\n✅ Normatividad institucional\n\n¿Qué necesitas saber?",
            "fuentes": []
        }
    
    try:
        # 🔥 MEJORA: Ejecutar pipeline completo
        rag_output = retrieve_and_rerank({"question": query.prompt})
        respuesta = (rag_prompt | llm | StrOutputParser()).invoke(rag_output)
        
        # Extraer fuentes únicas
        fuentes = list(set(
            os.path.basename(doc.metadata.get("source", "N/A")) 
            for doc in rag_output["source_documents"]
        ))
        
        print(f"\n✅ RESPUESTA: {respuesta[:300]}...")
        print(f"📎 FUENTES: {fuentes}\n")
        
        # 🔥 MEJORA: Validación más robusta
        respuesta_lower = respuesta.lower()
        indicadores_no_info = [
            "no encontré",
            "no tengo",
            "no está disponible",
            "no puedo proporcionar",
            "no hay información",
            "no se encuentra"
        ]
        
        tiene_info_valida = not any(ind in respuesta_lower for ind in indicadores_no_info)
        es_respuesta_corta = len(respuesta.strip()) < 15
        
        if not tiene_info_valida or es_respuesta_corta:
            return {
                "respuesta": "No encontré información específica sobre tu consulta en los documentos. Te recomiendo:\n• Verificar si usaste el término correcto\n• Contactar a Servicios Escolares\n• Revisar el sitio web oficial del TecNM Tlajomulco",
                "fuentes": []
            }
        
        return {
            "respuesta": respuesta,
            "fuentes": fuentes
        }
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "respuesta": "Ocurrió un error al procesar tu consulta. Por favor, intenta reformular tu pregunta de otra manera.",
            "fuentes": []
        }

@app.get("/")
async def root():
    return {
        "mensaje": "API del Chatbot TecNM Tlajomulco",
        "version": "3.0 - Mejorado",
        "estado": "✅ Operativo",
        "endpoints": {
            "/chat": "POST - Envía preguntas al chatbot",
            "/": "GET - Información de la API"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)