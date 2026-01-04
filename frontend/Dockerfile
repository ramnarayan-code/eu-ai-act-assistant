FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gradio_agent.py .
COPY rag_evaluator.py .

# Expose Gradio port
EXPOSE 7860

# Default environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "gradio_agent.py"]
