import gradio as gr
import requests

def interact_with_api(message, history):
    url = "http://127.0.0.1:8000/chat_with_agent"  # Local FastAPI URL
    payload = {"query": message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(response.json())
        return response.json()["response"]["blocks"][0]["text"]
    else:
        return "Error contacting the API"

# Define the Gradio ChatInterface
chat = gr.ChatInterface(
    interact_with_api,
    title="Chat with FastAPI",
    description="This Gradio interface interacts with a FastAPI POST endpoint."
)

chat.launch(share=False, inbrowser=True)