FROM python:3.13-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for building some python packages)
# RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_mcp_server.py .
COPY rag_knowledge_source.py .
COPY eu_ai_act_index ./eu_ai_act_index

# Default environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python", "rag_mcp_server.py"]
