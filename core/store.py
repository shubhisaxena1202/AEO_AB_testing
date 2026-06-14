import csv
from pathlib import Path


COLUMNS = ["query_id", "arm", "repeat", "recommended", "position", "accepted", "reasoning"]


def append_row(path: str, row: dict) -> None:
    p = Path(path)
    write_header = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_results(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
