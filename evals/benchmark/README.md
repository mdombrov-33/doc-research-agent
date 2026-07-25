# Evaluation Benchmark

This directory contains the fixed synthetic benchmark draft:

```text
benchmark/
├── fact_ledger.json
├── cases.jsonl
├── web_fixtures.json
└── corpus/
    ├── *.pdf
    ├── *.docx
    └── *.txt
```

`make eval-author` renders the PDF, DOCX, and TXT artifacts from `fact_ledger.json`.

`make eval-validate` checks schemas, cross-references, artifact presence, and agreement between
ledger passages and rendered documents. Neither command calls an embedding model, generator, or
judge.

The current vertical slice contains six People policy documents and ten draft cases. It proves
the authoring path; it is not yet the accepted benchmark.
