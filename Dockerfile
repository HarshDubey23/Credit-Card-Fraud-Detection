FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY model ./model
ENV PORT=8000
CMD ["sh", "-c", "uvicorn fraud.serve:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
