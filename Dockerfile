FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# COPY . . 
COPY src/app/app_backend.py src/app/
# COPY src/app/templates/index.html src/app/templates/
COPY src/models/model.py src/models/
COPY src/models/evaluate.py src/models/
COPY src/models/predict.py src/models/
COPY src/models/utils.py src/models/ 
COPY data/outputs/faiss/faiss.bin data/outputs/faiss/
COPY data/outputs/model-weights/8-11/coffee_model_epoch_11_semi_hard_3.pth data/outputs/model-weights/8-11/
COPY data/outputs/processed/preprocessed_data.csv data/outputs/processed/

EXPOSE 5000

ENV FLASK_APP=src.app.app_backend
ENV FLASK_DEBUG=1

# CMD ["flask", "run", "--host=0.0.0.0", "--port=$PORT", "--no-reload"]
CMD flask run --host=0.0.0.0 --port=$PORT --no-reload