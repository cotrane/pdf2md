"""Unit tests for the evaluate module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# Fix the import path
from evaluate import (
    calculate_levenshtein_metrics,
    calculate_rouge_scores,
    calculate_similarity,
    calculate_word_overlap,
    create_word_overlap_heatmap,
    evaluate_all_files,
    evaluate_files,
    find_files,
    preprocess_markdown,
)


@pytest.mark.unit
def test_preprocess_markdown() -> None:
    """Test removing markdown formatting from text."""
    markdown_text = """
# Header 1
## Header 2

This is **bold** and *italic* text.

- List item 1
- List item 2

1. Numbered item 1
2. Numbered item 2

> Blockquote

```python
def test():
    pass
```

[Link](https://example.com)

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

$$E = mc^2$$

---
    """

    expected = (
        "Header 1 Header 2 "
        "This is bold and italic text "
        "List item 1 List item 2 "
        "Numbered item 1 Numbered item 2 "
        "Blockquote "
        "python def test pass "
        "Link "
        "Header 1 Header 2Cell 1 Cell 2 Cell 3 Cell 4"
    )
    result = preprocess_markdown(markdown_text)

    # We normalize spaces to avoid test failures due to whitespace differences
    expected = " ".join(expected.split())
    result = " ".join(result.split())

    assert result == expected


@pytest.mark.unit
def test_calculate_similarity() -> None:
    """Test calculating similarity between two texts."""
    text1 = "This is a test sentence for similarity comparison"
    text2 = "This is a test phrase for similarity comparison"

    similarity = calculate_similarity(text1, text2)
    assert 0 <= similarity <= 1
    assert round(similarity, 2) == 0.75  # Should be high for similar texts


@pytest.mark.unit
def test_calculate_similarity_empty() -> None:
    """Test calculating similarity with empty texts."""
    text1 = ""
    text2 = "This is not empty"

    similarity = calculate_similarity(text1, text2)
    assert similarity == 0.0


@pytest.mark.unit
def test_calculate_word_overlap() -> None:
    """Test calculating word overlap between two texts."""
    text1 = "apple banana cherry date"
    text2 = "apple banana fig grape"

    overlap_ratio, words1, words2 = calculate_word_overlap(text1, text2)

    assert overlap_ratio == 2 / 6  # 2 common words out of 6 unique words
    assert words1 == {"apple", "banana", "cherry", "date"}
    assert words2 == {"apple", "banana", "fig", "grape"}


@pytest.mark.unit
def test_calculate_levenshtein_metrics() -> None:
    """Test calculating Levenshtein metrics between two texts."""
    text1 = "kitten"
    text2 = "sitting"

    levenshtein_dist, levenshtein_ratio = calculate_levenshtein_metrics(text1, text2)

    assert levenshtein_dist == 3  # 3 operations to transform kitten to sitting
    assert 0 <= levenshtein_ratio <= 1


@pytest.mark.unit
def test_calculate_rouge_scores() -> None:
    """Test calculating ROUGE scores between two texts."""
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The fast brown fox jumps over the lazy dog"

    rouge_scores = calculate_rouge_scores(text1, text2)

    assert "rouge-1" in rouge_scores
    assert "rouge-2" in rouge_scores
    assert "rouge-l" in rouge_scores
    assert all(0 <= score <= 1 for score in rouge_scores.values())


@pytest.mark.unit
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.close")
@patch("seaborn.heatmap")
def test_create_word_overlap_heatmap(
    mock_heatmap: MagicMock, mock_close: MagicMock, mock_figure: MagicMock, mock_savefig: MagicMock
) -> None:
    """Test creating a heatmap visualization of overlap ratios."""
    files = ["file1.md", "file2.md", "file3.md"]
    overlap_ratios = [
        {"file1": "file1.md", "file2": "file2.md", "value": 0.5},
        {"file1": "file1.md", "file2": "file3.md", "value": 0.3},
        {"file1": "file2.md", "file2": "file3.md", "value": 0.7},
    ]
    labels = ["model1", "model2", "model3"]

    create_word_overlap_heatmap(files, overlap_ratios, labels, "test_heatmap.png")

    assert mock_figure.call_count == 5
    mock_heatmap.assert_called_once()
    mock_savefig.assert_called_once_with("test_heatmap.png", bbox_inches="tight", dpi=300)
    mock_close.assert_called_once()


@pytest.mark.unit
@patch("glob.glob")
def test_find_files(mock_glob: MagicMock) -> None:
    """Test finding files with the given filestem and parsers."""
    mock_glob.side_effect = lambda pattern: {
        "output/test_anthropic*.md": ["output/test_anthropic.md"],
        "output/test_googleai*.md": [
            "output/test_googleai.md",
            "output/test_googleai_gemini-2.0-flash.md",
        ],
        "output/test_openai*.md": ["output/test_openai.md"],
    }[pattern]

    files = find_files("test", ["anthropic", "googleai", "openai"])

    assert len(files) == 4
    assert "output/test_anthropic.md" in files
    assert "output/test_googleai.md" in files
    assert "output/test_googleai_gemini-2.0-flash.md" in files
    assert "output/test_openai.md" in files


@pytest.mark.unit
@patch("glob.glob")
def test_find_files_no_matches(mock_glob: MagicMock) -> None:
    """Test finding files with no matches."""
    mock_glob.return_value = []

    with pytest.raises(ValueError, match="No files found for filestem"):
        find_files("nonexistent", ["anthropic"])


@pytest.mark.unit
@patch("builtins.open", new_callable=mock_open, read_data="# Test\nThis is a test document.")
@patch("evaluate.preprocess_markdown", return_value="This is a test document")
@patch("evaluate.calculate_similarity", return_value=0.95)
@patch(
    "evaluate.calculate_word_overlap",
    return_value=(
        0.8,
        {"this", "is", "a", "test", "document"},
        {"this", "is", "a", "test", "document"},
    ),
)
@patch("evaluate.calculate_levenshtein_metrics", return_value=(5, 0.9))
@patch(
    "evaluate.calculate_rouge_scores",
    return_value={"rouge-1": 0.9, "rouge-2": 0.85, "rouge-l": 0.88},
)
def test_evaluate_files(
    mock_rouge: MagicMock,
    mock_levenshtein: MagicMock,
    mock_overlap: MagicMock,
    mock_similarity: MagicMock,
    mock_preprocess: MagicMock,
    mock_file: MagicMock,
) -> None:
    """Test evaluating similarity between two markdown files."""
    file1 = "test_file1.md"
    file2 = "test_file2.md"

    with patch("os.path.basename") as mock_basename:
        mock_basename.side_effect = lambda f: f

        cosine_sim, overlap_ratio, levenshtein_ratio, rouge_scores = evaluate_files(file1, file2)

        assert cosine_sim == 0.95
        assert overlap_ratio == 0.8
        assert levenshtein_ratio == 0.9
        assert rouge_scores["rouge-1"] == 0.9
        assert rouge_scores["rouge-2"] == 0.85
        assert rouge_scores["rouge-l"] == 0.88


@pytest.mark.unit
@patch("builtins.open")
def test_evaluate_files_file_not_found(mock_open: MagicMock) -> None:
    """Test evaluating files with a non-existent file."""
    mock_open.side_effect = FileNotFoundError("File not found")

    with pytest.raises(FileNotFoundError):
        evaluate_files("nonexistent1.md", "nonexistent2.md")


@pytest.mark.integration
@patch("evaluate.find_files")
@patch("evaluate.evaluate_files")
@patch("evaluate.create_word_overlap_heatmap")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate_all_files_integration(
    mock_file: MagicMock, mock_heatmap: MagicMock, mock_evaluate: MagicMock, mock_find: MagicMock
) -> None:
    """Test the full workflow of evaluating all files."""
    # Mock the file finding function
    mock_find.return_value = [
        "output/doc_anthropic.md",
        "output/doc_googleai.md",
        "output/doc_openai.md",
    ]

    # Mock the evaluation results
    mock_evaluate.return_value = (
        0.85,
        0.75,
        0.8,
        {"rouge-1": 0.8, "rouge-2": 0.7, "rouge-l": 0.75},
    )

    # Call the function being tested
    evaluate_all_files("doc", ["anthropic", "googleai", "openai"])

    # Check that the right functions were called
    mock_find.assert_called_once_with("doc", ["anthropic", "googleai", "openai"])

    # Should call evaluate_files for each pair of files (including self-comparisons)
    assert mock_evaluate.call_count == 6

    # Should create 6 heatmaps (one for each metric)
    assert mock_heatmap.call_count == 6

    # Should write to the CSV file
    mock_file.assert_called_once_with("output/doc_metrics.csv", "w", encoding="utf-8")
