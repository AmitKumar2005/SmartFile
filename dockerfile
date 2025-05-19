FROM python:3.11-slim

RUN apt-get update && apt-get install -y 
poppler-utils 
tesseract-ocr 
libtesseract-dev 
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app COPY . . RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]