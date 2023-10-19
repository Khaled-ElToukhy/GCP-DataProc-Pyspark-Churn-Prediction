FROM python:3.9

RUN pip install pandas

RUN pip install sqlalchemy psycopg2

COPY dataset.csv dataset.csv

COPY ingest.py ingest.py

ENTRYPOINT ["python","ingest.py"]






