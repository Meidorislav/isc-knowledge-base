from app.search import RrfHit


def format_context(hits: list[RrfHit]) -> str:
    blocks: list[str] = []
    for number, hit in enumerate(hits, start=1):
        heading = " > ".join(hit.heading_path)
        blocks.append(f"[{number}] {heading}\n{hit.text}")
    return "\n\n".join(blocks)
