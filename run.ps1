param(
    [int]$Port = 8000
)

python -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port $Port
