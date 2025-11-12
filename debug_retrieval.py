from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN ---
DB_FAISS_PATH = 'vectorstore/db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# La pregunta que quieres probar. Usa una pregunta que SABES que la respuesta está en los PDFs.
TU_PREGUNTA_DE_PRUEBA_AQUI = "¿Cuáles son los requisitos para el servicio social?"

# --- SCRIPT DE PRUEBA ---
def test_retrieval():
    print("Iniciando prueba de diagnóstico de retrieval...")

    # 1. Cargar el mismo modelo de embeddings que usa la aplicación
    print(f"Cargando el modelo de embeddings: {EMBEDDING_MODEL}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
        print("Modelo de embeddings cargado con éxito.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo de embeddings. {e}")
        return

    # 2. Cargar la base de datos vectorial existente
    print(f"Cargando la base de datos desde: {DB_FAISS_PATH}...")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Base de datos vectorial cargada con éxito.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar la base de datos. Asegúrate de haber ejecutado 'procesar_documentos.py' primero. {e}")
        return

    # 3. Realizar la búsqueda de similitud
    print(f"\nRealizando búsqueda para la pregunta: '{TU_PREGUNTA_DE_PRUEBA_AQUI}'")
    # Usamos similarity_search para ver los resultados directos
    # k=5 significa que buscamos los 5 fragmentos más relevantes
    results = db.similarity_search(TU_PREGUNTA_DE_PRUEBA_AQUI, k=5)

    # 4. Mostrar los resultados
    if not results:
        print("\n--- RESULTADO: ¡NO SE ENCONTRÓ NINGÚN DOCUMENTO! ---")
        print("Esto confirma que la búsqueda no está devolviendo nada.")
    else:
        print(f"\n--- RESULTADO: Se encontraron {len(results)} fragmentos ---")
        for i, doc in enumerate(results):
            print(f"\n--- Documento Relevante #{i+1} ---")
            print(f"Fuente: {doc.metadata.get('source', 'N/A')}")
            print("Contenido del fragmento:")
            print(doc.page_content)
            print("---------------------------------")

if __name__ == "__main__":
    test_retrieval()