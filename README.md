# chatbotLLM

Esta es una App en Python que crea un chatbot que utiliza la AI para responder preguntas sobre temas del Tecnologico de Tlajomulco a docentes y estudiantes.
Esta App funciona en Local, no utiliza las AI de la nube.

# Requerimientos
Hardware
Terjeta de Video GeForce 3070 con 16 RAM o superior

Software
LM Studio ejecutando un LLM
Developer Status: Iniciado
Los modelos con los que yo probe la App
<img width="733" height="305" alt="image" src="https://github.com/user-attachments/assets/449cff25-8a3e-4e78-8309-a3b19f21f4ee" />


Python 3.12.7

# Ejecutar
pip install -r requirements.txt

python procesar_doctos.py

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# En el navegador
index.html
