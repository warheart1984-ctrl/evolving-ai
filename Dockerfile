FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY app ./app
COPY constitution ./constitution
COPY evaluations ./evaluations
COPY prompts ./prompts

RUN pip install --no-cache-dir "fastapi>=0.100" "uvicorn[standard]" "pydantic>=2.5"

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
