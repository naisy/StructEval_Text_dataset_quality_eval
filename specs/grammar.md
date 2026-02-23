# Grammar validation

## Goal

Detect records whose *output* is not syntactically valid for its declared output format.

## Supported formats

- JSON: `json.loads`
- YAML: `yaml.safe_load` (+ optional 2-space indent style rule)
- TOML: `tomllib.loads` (Python 3.11+)
- XML: `xml.etree.ElementTree.fromstring`
- CSV: strict-ish parsing where all rows match header width

## Payload extraction

Before parsing, attempt to extract the likely payload:

1. If fenced code blocks exist, use the **last** code block.
2. If a marker line exists (`Output:` / `Answer:` / `Final:`), take text after it.
3. For JSON/XML, take from the first `{`/`[` or `<`.
4. Otherwise, strip only ``` fence lines.
