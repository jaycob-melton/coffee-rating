FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

EXPOSE 5000

ENV FLASK_APP=src.app.app_backend
ENV FLASK_DEBUG=1

# CMD ["flask", "run", "--host=0.0.0.0", "--port=$PORT", "--no-reload"]
CMD flask run --host=0.0.0.0 --port=$PORT --no-reload