# chatbotLLM

Esta es una App en Python
Tiene 3 funciones principales:
1. procesar_doctos.py: Procesa todos los pdfs de la carpeta normatividad y genera una base de datos vectorial que se utiliza posteriormente para la busqueda de informacion.
2. main.py: es el programa que se ejecuta como API y recibe las peticiones del chatbot.
3. index.html, chat-widget.js, chat-widget.css son el chatbot que se ejecuta en el navegador.

Descripcion: crea un chatbot que utiliza la AI para responder preguntas sobre temas del Tecnologico de Tlajomulco a docentes y estudiantes.
Esta App funciona en Local, no utiliza las AI de la nube.

## Requerimientos
Hardware
Terjeta de Video GeForce 3070 con 16 RAM o superior

Software
LM Studio ejecutando un LLM
Developer Status: Iniciado

Los modelos con los que yo probe la App:

Yi 34B Chat Q4_K_S TheBloke 30B llama GGUF 18.20 GB

Claude 3.7 Sonnet Reasoning Gemma3 12B Q8_0  reedmayhew 12B gemma3 GGUF 11.65 GB

Llama 2 13B Chat Q3_K_S TheBloke 13B llama GGUF 5.27 GB

Llama 3.3 70B Instruct Q4_K_S bartowski 70B llama GGUF 37.58 GB

Llama 4 Scout 17B 16E Instruct Q3_K_S unsloth 17B llama4 GGUF 45.16 GB

<img width="733" height="305" alt="image" src="https://github.com/user-attachments/assets/449cff25-8a3e-4e78-8309-a3b19f21f4ee" />


Python 3.12.7

## Ejecutar
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## En: https://huggingface.co/
1. log in or create account
2. Settings
3. Access Tokens
4. Create New token
5. Copy token

#### Pegar el token cuando lo solicite la siguiente linea
huggingface-cli login

# Ejecutar
```bash
python procesar_doctos.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

# En el navegador
index.html

# Cada que hay cambios en los documentos pdf es necesario
```bash
rm -rf vectorstore
python procesar_doctos.py
```