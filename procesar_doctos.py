import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

PDFS_PATH = "normatividad"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
BATCH_SIZE = 32

def cargar_documentos(path):
    print(f"Cargando documentos de forma recursiva desde {path}...")
    loader = PyPDFDirectoryLoader(path, recursive=True, silent_errors=True)
    documents = loader.load()
    if not documents:
        print("¡Advertencia! No se encontraron documentos PDF.")
        return []
    print(f"Se cargaron {len(documents)} páginas/documentos.")
    return documents

def dividir_documentos(documents):
    if not documents:
        return []
    print("Dividiendo documentos en trozos optimizados...")
    
    # 🔥 MEJORA 1: Chunks más grandes para mantener contexto completo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Aumentado de 600 a 1200
        chunk_overlap=200,  # Aumentado para mejor continuidad
        separators=["\n\n", "\n", ". ", " ", ""],  # Mejor separación semántica
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Se crearon {len(chunks)} trozos de texto.")
    return chunks

def crear_base_de_datos_vectorial(chunks):
    if not chunks:
        return

    print(f"Inicializando embeddings con el modelo: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32  # 🔥 MEJORA 2: Procesar en lotes eficientemente
        }
    )

    print("Creando la base de datos vectorial...")
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    primer_lote_texts = texts[:BATCH_SIZE]
    primer_lote_metadatas = metadatas[:BATCH_SIZE]
    db = FAISS.from_texts(primer_lote_texts, embeddings, metadatas=primer_lote_metadatas)
    
    for i in tqdm(range(BATCH_SIZE, len(texts), BATCH_SIZE), desc="Procesando lotes"):
        lote_actual_texts = texts[i:i + BATCH_SIZE]
        lote_actual_metadatas = metadatas[i:i + BATCH_SIZE]
        db.add_texts(texts=lote_actual_texts, metadatas=lote_actual_metadatas)

    db.save_local(DB_FAISS_PATH)
    print(f"✅ Base de datos vectorial guardada en {DB_FAISS_PATH}")

if __name__ == "__main__":
    documentos = cargar_documentos(PDFS_PATH)
    trozos_de_texto = dividir_documentos(documentos)
    crear_base_de_datos_vectorial(trozos_de_texto)