"""Bulk import of existing documents (one-time wiki migration).

    python -m app.ingestion.cli ./wiki_export --owner "Иванов И.И." [--category Регламенты]

Idempotent: files already uploaded (same content hash) are skipped, so it's safe to re-run.
"""

import argparse
from pathlib import Path

from app.db import SessionLocal
from app.ingestion.parsing import SUPPORTED_FORMATS, UnsupportedFormatError
from app.ingestion.service import DuplicateDocumentError, submit_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a folder of documents into the knowledge base")
    parser.add_argument("path", type=Path, help="folder (searched recursively) or a single file")
    parser.add_argument("--owner", required=True, help="who is responsible for these documents")
    parser.add_argument("--category")
    args = parser.parse_args()

    files = [args.path] if args.path.is_file() else sorted(p for p in args.path.rglob("*") if p.is_file())
    added = skipped = failed = 0

    with SessionLocal() as session:
        for path in files:
            if path.suffix.lower().lstrip(".") not in SUPPORTED_FORMATS | {"markdown"}:
                continue
            try:
                doc = submit_document(
                    session,
                    filename=path.name,
                    raw=path.read_bytes(),
                    owner=args.owner,
                    category=args.category,
                )
                added += 1
                print(f"+ {path}  ->  {doc.id}")
            except DuplicateDocumentError as e:
                session.rollback()
                skipped += 1
                print(f"= {path}  (уже есть: {e.document_id})")
            except UnsupportedFormatError as e:
                session.rollback()
                failed += 1
                print(f"! {path}  ({e})")

    print(f"\nДобавлено: {added}, пропущено дублей: {skipped}, ошибок: {failed}")
    print("Документы поставлены в очередь; индексирует их воркер (python -m app.ingestion.worker).")


if __name__ == "__main__":
    main()
