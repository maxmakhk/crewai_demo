FROM python:3.11-slim
RUN pip install -r requirements.txt
WORKDIR /app
COPY . /app
EXPOSE 8000
CMD ["python", "run.py"]
