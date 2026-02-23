from dataset_eval.final_output import extract_final_output


def test_output_marker_wins_over_preamble():
    text = 'Approach: blah\n\nOutput:\n{"a": 1}'
    assert extract_final_output(text, "JSON") == '{"a": 1}'


def test_last_fence_is_used():
    text = "```json\n{\"a\": 1}\n```\n\nnoise\n```json\n{\"b\": 2}\n```"
    assert extract_final_output(text, "JSON") == '{"b": 2}'


def test_csv_strips_leading_explanation():
    text = "Here is the CSV you asked for:\ncol1,col2\n1,2\n"
    assert extract_final_output(text, "CSV") == "col1,col2\n1,2"


def test_fallback_strip():
    assert extract_final_output("  hello  ", None) == "hello"
