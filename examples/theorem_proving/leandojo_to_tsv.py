r"""Convert a LeanDojo benchmark split, already on disk, into this example's TSV files.

This script never downloads anything; fetch the benchmark yourself first
(see README.md). It reads:

- ``corpus.jsonl``: one file per line, ``premises`` with ``full_name`` and ``code``
- a split file such as ``random/test.json``: theorems with ``traced_tactics``,
  each with ``state_before`` and ``annotated_tactic = [tactic, [{full_name, ...}]]``

and writes ``lemmas.tsv`` (no dependency column; LeanDojo does not provide
premise-to-premise edges) and ``proof_steps.tsv`` for steps that use at
least one premise. Then run::

    python premise_retrieval.py --no-demo \
        --lemmas OUT/lemmas.tsv --steps OUT/proof_steps.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def one_line(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, help="path to corpus.jsonl")
    parser.add_argument("split", type=Path, help="path to e.g. random/test.json")
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names: set[str] = set()
    with (
        args.corpus.open(encoding="utf-8") as src,
        (args.out / "lemmas.tsv").open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            record = json.loads(line)
            for premise in record.get("premises", []):
                name = premise.get("full_name")
                if not name or name in names:
                    continue
                names.add(name)
                code = one_line(premise.get("code", ""))
                dst.write(f"{name}\t{record.get('path', '')}\t{code}\t\n")

    steps = 0
    theorems = json.loads(args.split.read_text(encoding="utf-8"))
    with (args.out / "proof_steps.tsv").open("w", encoding="utf-8") as dst:
        for theorem in theorems:
            for tactic in theorem.get("traced_tactics", []):
                annotated = tactic.get("annotated_tactic") or ["", []]
                useful = sorted(
                    {
                        p["full_name"]
                        for p in annotated[1]
                        if p.get("full_name") in names
                    }
                )
                if useful and steps < args.max_steps:
                    dst.write(
                        f"{one_line(tactic['state_before'])}\t{','.join(useful)}\n"
                    )
                    steps += 1
    print(f"wrote {len(names)} premises and {steps} proof steps to {args.out}")


if __name__ == "__main__":
    main()
