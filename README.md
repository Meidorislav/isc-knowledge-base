# ISC Knowledge Base

Import Markdown into PostgreSQL, generate embeddings with the trainable NumPy encoder, and store them with pgvector. The repository includes a small synthetic benchmark for running this workflow from a clean checkout.

## Run the demo

From the repository root, install the locked dependencies and start PostgreSQL.

```sh
uv sync --locked
docker compose up -d --wait
```

Apply the migrations once on a fresh development database.

```sh
cat db/migrations/*.up.sql | docker compose exec -T postgres sh -ec \
    'export PGPASSWORD="$POSTGRES_PASSWORD"; exec psql -X -h 127.0.0.1 -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Train the encoder on the bundled fixture and import the demo project. Passing `--checkpoint` publishes its fragments and vectors in one transaction.

```sh
uv run --locked experiments/encoder_benchmark.py --model-out /tmp/isc-demo-encoder.npz
uv run --locked python -m isc_knowledge_base.postgres_import \
    experiments/fixtures/benchmark/corpus \
    --project-id demo \
    --checkpoint /tmp/isc-demo-encoder.npz
```

The benchmark accepts a custom directory containing `corpus/` and `questions.json` as its first argument.

Search the demo project with the same checkpoint. The JSON results include the fragment text, score, and source lines.

```sh
uv run --locked python -m isc_knowledge_base.semantic_search \
    /tmp/isc-demo-encoder.npz --project-id demo --query "hotel invoice" --limit 5
```

## Search over HTTP

Start the API with the indexed checkpoint.

```sh
env ENCODER_CHECKPOINT_PATH=/tmp/isc-demo-encoder.npz \
    uv run --locked uvicorn isc_knowledge_base.api:app --host 127.0.0.1
```

From another terminal, send a JSON request to `POST /search`.

```sh
curl -sS http://127.0.0.1:8000/search \
    -H 'Content-Type: application/json' \
    -d '{"project_id":"demo","query":"hotel invoice","limit":5}'
```

The API returns the same array of fragments as the CLI and sets `Cache-Control: no-store`. The limit defaults to 5 and accepts integers from 1 to 100.

## Run tests

```sh
uv run --locked python -m unittest discover -s tests -v
```

Set `POSTGRES_TEST_DSN` to a development PostgreSQL connection string to include the integration tests. They create and remove a temporary schema.
