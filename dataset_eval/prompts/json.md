JSON format checklist (the answer is intended to be JSON):
- Output must be JSON text only (no explanation, no Markdown).
- Must be strict JSON (RFC 8259):
  - No trailing commas.
  - Double quotes for strings/keys.
  - No NaN/Infinity.
- Must match the task requirement (right top-level type/object/array, required keys present, no forbidden keys).
- If instruction requests “only JSON” or a specific schema/shape, treat any extra text or mismatched structure as incorrect.
