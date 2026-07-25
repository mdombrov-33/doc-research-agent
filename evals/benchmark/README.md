# Evaluation Benchmark

This directory will contain the fixed synthetic benchmark built in milestone 2:

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

`make eval-validate` checks schemas, cross-references, and corpus artifacts without calling an
embedding model, generator, or judge. The command is expected to report missing benchmark files
until the first draft is generated.
