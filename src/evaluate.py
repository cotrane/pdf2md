#!/usr/bin/env python3
"""Script to evaluate similarity between markdown files given a filestem and a list of parsers."""

import argparse
import csv
import glob
import itertools
import logging
import os
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Levenshtein import distance, ratio
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logging import setup_logging  # pylint: disable=no-name-in-module

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def preprocess_markdown(text: str) -> str:
    """Remove markdown formatting and clean the text.

    Args:
        text: Input markdown text.

    Returns:
        Cleaned text without markdown formatting.
    """
    # Remove headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Remove bold and italic - handle both single and double markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold with **
    text = re.sub(r"__([^_]+)__", r"\1", text)  # Bold with __
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic with *
    text = re.sub(r"_([^_]+)_", r"\1", text)  # Italic with _
    text = re.sub(r"[~`]", "", text)  # Remove remaining markdown markers

    # Remove links but keep link text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove lists
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove blockquotes
    text = re.sub(r"^\s*>\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Handle tables - preserve cell contents while removing markdown markers
    # First, remove table separators (the line with dashes)
    text = re.sub(r"^\s*\|[-:|\s]+\|\s*$", "", text, flags=re.MULTILINE)
    # Remove line breaks in tables
    text = re.sub(r"<br>", " ", text)
    # Then, remove the pipe characters but keep cell contents, with exactly one
    # space around each cell
    text = re.sub(
        r"^\s*\|(.*?)\|\s*$",
        lambda m: " ".join(cell.strip() for cell in m.group(1).split("|")),
        text,
        flags=re.MULTILINE,
    )
    # Clean up any remaining pipe characters in the middle of lines
    text = re.sub(r"\|", " ", text)
    # Clean up multiple spaces between cells
    text = re.sub(r"\s{2,}", " ", text)

    # Remove mathematical formulas
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)  # Display math
    text = re.sub(r"\$[^$]+\$", "", text)  # Inline math
    text = re.sub(r"\\\(.*?\\\)", "", text)  # LaTeX inline math
    text = re.sub(r"\\\[.*?\\\]", "", text)  # LaTeX display math
    text = re.sub(r"\\mathrm{[^}]+}", "", text)  # LaTeX \mathrm commands
    text = re.sub(r"\\text{[^}]+}", "", text)  # LaTeX \text commands
    text = re.sub(r"\\frac{[^}]+}{[^}]+}", "", text)  # LaTeX fractions
    text = re.sub(r"\\sqrt{[^}]+}", "", text)  # LaTeX square roots
    text = re.sub(r"\\sum", "", text)  # LaTeX sum symbol
    text = re.sub(r"\\int", "", text)  # LaTeX integral symbol
    text = re.sub(r"\\infty", "", text)  # LaTeX infinity symbol
    text = re.sub(
        r"\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\zeta|\\eta|\\theta|\\iota|\\kappa|\\lambda|"
        r"\\mu|\\nu|\\xi|\\pi|\\rho|\\sigma|\\tau|\\upsilon|\\phi|\\chi|\\psi|\\omega",
        "",
        text,
    )  # Greek letters
    text = re.sub(
        r"\\Gamma|\\Delta|\\Theta|\\Lambda|\\Xi|\\Pi|\\Sigma|\\Phi|\\Psi|\\Omega",
        "",
        text,
    )  # Capital Greek letters

    # Remove special characters and symbols
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # Any remaining LaTeX commands
    text = re.sub(r"&[a-zA-Z]+;", "", text)  # HTML entities
    text = re.sub(r"\\[^a-zA-Z]", "", text)  # Escaped special characters
    text = re.sub(r"[^\w\s]", " ", text)  # Replace any remaining special characters with space

    # Remove page numbers and section markers
    text = re.sub(r"######\s+Page\s+\d+", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^\s*[*_]\s*\([^)]+\)\s*[*_]\s*$", "", text, flags=re.MULTILINE
    )  # Italicized parenthetical text
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # Standalone page numbers
    text = re.sub(
        r"^\s*Table of Contents\s*$", "", text, flags=re.MULTILINE
    )  # Table of Contents headers
    text = re.sub(r"^\s*\[.*?\]\s*$", "", text, flags=re.MULTILINE)  # Image references
    text = re.sub(r"^\s*\[.*?\]\(.*?\)\s*$", "", text, flags=re.MULTILINE)  # Image links
    text = re.sub(r"^\s*\[.*?\]\s*$", "", text, flags=re.MULTILINE)  # Footnote references
    text = re.sub(r"^\s*\[.*?\]:\s*.*$", "", text, flags=re.MULTILINE)  # Footnote definitions

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using TF-IDF and cosine similarity.

    Args:
        text1: First text to compare.
        text2: Second text to compare.

    Returns:
        Similarity score between 0 and 1.
    """
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        # If one of the texts is empty or contains no valid words
        return 0.0


def calculate_word_overlap(text1: str, text2: str) -> tuple[float, set[str], set[str]]:
    """Calculate word overlap between two texts.

    Args:
        text1: First text to compare.
        text2: Second text to compare.

    Returns:
        Tuple containing (overlap ratio, words in text1, words in text2).
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    overlap = words1.intersection(words2)
    total = words1.union(words2)

    overlap_ratio = len(overlap) / len(total) if total else 0.0

    return overlap_ratio, words1, words2


def calculate_levenshtein_metrics(text1: str, text2: str) -> tuple[int, float]:
    """Calculate Levenshtein distance and ratio between two texts.

    Args:
        text1: First text to compare.
        text2: Second text to compare.

    Returns:
        Tuple containing (Levenshtein distance, Levenshtein ratio).
    """
    return distance(text1, text2), ratio(text1, text2)


def calculate_rouge_scores(text1: str, text2: str) -> dict[str, float]:
    """Calculate ROUGE scores between two texts.

    Args:
        text1: First text to compare.
        text2: Second text to compare.

    Returns:
        Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    # Set a higher recursion limit
    sys.setrecursionlimit(10000)

    # Limit text length to avoid recursion issues
    max_length = 10000
    if len(text1) > max_length:
        text1 = text1[:max_length]
    if len(text2) > max_length:
        text2 = text2[:max_length]

    rouge = Rouge()
    try:
        scores = rouge.get_scores(text1, text2, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
    except (ValueError, RecursionError) as e:
        logger.warning(f"Error calculating ROUGE scores: {e}")
        # If one of the texts is empty or contains no valid words
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    finally:
        # Reset recursion limit to default
        sys.setrecursionlimit(1000)


def create_word_overlap_heatmap(
    files: list[str],
    overlap_ratios: list[dict[str, str | float]],
    labels: list[str],
    output_file: str = "overlap_heatmap.png",
    title: str = "Word Overlap Ratios Heatmap",
    label: str = "Overlap Ratio",
    cmap: str = "RdYlBu_r",
) -> None:
    """Create a heatmap visualization of overlap ratios between all pairs of files.

    Args:
        files: List of file paths.
        overlap_ratios: List of dictionaries containing file pairs and their overlap ratios.
            Each dictionary has keys "file1" (str), "file2" (str), and "overlap_ratio" (float).
        output_file: Path to save the heatmap image.
        title: Title of the heatmap.
        label: Label of the heatmap.
        cmap: Colormap of the heatmap.
    """
    plt.style.use("fivethirtyeight")

    # Create a matrix of overlap ratios
    n_files = len(files)
    matrix = np.zeros((n_files, n_files))

    # Create a mapping of file paths to indices
    file_to_idx = {os.path.basename(file): idx for idx, file in enumerate(files)}

    # Fill the matrix with overlap ratios
    for ratio_data in overlap_ratios:
        i = file_to_idx[str(ratio_data["file1"])]
        j = file_to_idx[str(ratio_data["file2"])]
        matrix[i, j] = float(ratio_data["value"])
        matrix[j, i] = float(ratio_data["value"])  # Make the matrix symmetric

    # Create heatmap
    plt.figure(figsize=(12, 12))  # Increased figure size for better readability
    ax = sns.heatmap(  # pylint: disable=unused-variable
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={
            "label": label,
            "shrink": 0.5,  # Make colorbar even smaller (50% of original size)
            "aspect": 30,  # Make colorbar thinner
            "pad": 0.02,  # Reduce padding
        },
        square=True,  # Make cells square
        annot=True,  # Show values in cells
        fmt=".2f",  # Format values to 2 decimal places
        annot_kws={"size": 10},  # Adjust annotation size
    )

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    # Rotate y-axis labels
    plt.yticks(rotation=0)

    plt.title(title, pad=30, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=300)  # Increased DPI for better quality
    plt.close()
    logger.debug(f"\nHeatmap saved as: {output_file}")


def find_files(filestem: str, parsers: list[str]) -> list[str]:
    """Find all files with the given filestem and parsers.

    Args:
        filestem: The base name of the files to compare.
        parsers: List of space-separated parsers to take into account.
    """
    files = []
    for parser in parsers:
        pattern = f"output/{filestem}_{parser}*.md"
        matching_files = glob.glob(pattern)
        files.extend(matching_files)
    if len(files) == 0:
        logger.exception(f"Error: No files found for filestem {filestem} and parsers {parsers}")
        raise ValueError(f"No files found for filestem {filestem} and parsers {parsers}")
    logger.debug(f"Found {len(files)} files for filestem {filestem} and parsers {parsers}")
    logger.debug(files)
    return files


def evaluate_files(
    file1: str, file2: str, preprocess: bool = True
) -> tuple[float, float, float, dict[str, float]]:
    """Evaluate similarity between two markdown files.

    Args:
        file1: Path to first markdown file.
        file2: Path to second markdown file.
        preprocess: Whether to preprocess the text and remove markdown formatting.
            Default is True.

    Raises:
        FileNotFoundError: If the files are not found.
        Exception: If the files cannot be read.

    Returns:
        Tuple containing (cosine similarity, word overlap ratio, levenshtein ratio, rouge scores).
    """
    # Read files
    try:
        with open(file1, "r", encoding="utf-8") as f:
            text1 = f.read()
        with open(file2, "r", encoding="utf-8") as f:
            text2 = f.read()
    except FileNotFoundError as e:
        logger.exception(f"Error: Could not find file - {e}")
        raise
    except Exception as e:
        logger.exception(f"Error reading files: {e}")
        raise

    # Preprocess markdown
    if preprocess:
        clean_text1 = preprocess_markdown(text1)
        clean_text2 = preprocess_markdown(text2)
    else:
        clean_text1 = text1
        clean_text2 = text2

    # Calculate similarities
    cosine_sim = calculate_similarity(clean_text1, clean_text2)
    overlap_ratio, words1, words2 = calculate_word_overlap(clean_text1, clean_text2)
    _, levenshtein_ratio = calculate_levenshtein_metrics(clean_text1, clean_text2)
    rouge_scores = calculate_rouge_scores(clean_text1, clean_text2)

    # logger.info results
    model1 = os.path.basename(file1).split("_", 1)[1].removesuffix(".md")
    model2 = os.path.basename(file2).split("_", 1)[1].removesuffix(".md")
    logger.info(f"Evaluation Results: {model1} vs {model2}")
    logger.info("-" * 50)
    logger.info(f"Cosine Similarity: {cosine_sim:.4f}")
    logger.info(f"Word Overlap Ratio: {overlap_ratio:.4f}")
    logger.info(f"Levenshtein Ratio: {levenshtein_ratio:.4f}")
    logger.info(f"ROUGE-1 Score: {rouge_scores['rouge-1']:.4f}")
    logger.info(f"ROUGE-2 Score: {rouge_scores['rouge-2']:.4f}")
    logger.info(f"ROUGE-L Score: {rouge_scores['rouge-l']:.4f}\n")
    logger.info(f"Words in {model1}: {len(words1)}")
    logger.info(f"Words in {model2}: {len(words2)}")
    logger.info(f"Common words: {len(words1.intersection(words2))}")
    logger.info(f"Unique to {model1}: {len(words1 - words2)}")
    logger.info(f"Unique to {model2}: {len(words2 - words1)}")

    # log unique words
    logger.debug(f"\nUnique words in {model1}:")
    logger.debug("-" * 50)
    logger.debug(sorted(words1 - words2))
    logger.debug(f"\nUnique words in {model2}:")
    logger.debug("-" * 50)
    logger.debug(sorted(words2 - words1))

    print("")

    return cosine_sim, overlap_ratio, levenshtein_ratio, rouge_scores


# pylint: disable=too-many-locals
def evaluate_all_files(filestem: str, parsers: list[str], remove_markdown: bool = True) -> None:
    """Evaluate similarity between all pairs of markdown files.

    Args:
        filestem: The base name of the files to compare.
        parsers: List of space-separated parsers to take into account.
        remove_markdown: Whether to remove markdown formatting from the files.
            Default is True.
    """
    files = find_files(filestem, parsers)
    suffix = " with Markdown" if not remove_markdown else ""

    # Calculate all metrics in a single pass
    metrics_data: list[dict[str, Any]] = []
    for file1, file2 in itertools.combinations_with_replacement(files, 2):
        cosine_sim, overlap_ratio, levenshtein_ratio, rouge_scores = evaluate_files(
            file1, file2, preprocess=remove_markdown
        )
        metrics_data.append(
            {
                "files": (os.path.basename(file1), os.path.basename(file2)),
                "word_overlap": overlap_ratio,
                "levenshtein": levenshtein_ratio,
                "cosine": cosine_sim,
                "rouge": rouge_scores,
            }
        )

    # Define metric configurations
    metric_configs: list[dict[str, str]] = [
        {
            "name": "word_overlap",
            "title": f"Word Overlap Ratios Heatmap{suffix}",
            "label": "Overlap Ratio",
        },
        {
            "name": "levenshtein",
            "title": f"Levenshtein Ratios Heatmap{suffix}",
            "label": "Levenshtein Ratio",
        },
        {"name": "cosine", "title": f"Cosine Ratios Heatmap{suffix}", "label": "Cosine Ratio"},
    ]

    # Add ROUGE metrics
    for rouge_type in ["rouge-1", "rouge-2", "rouge-l"]:
        metric_configs.append(
            {
                "name": rouge_type,
                "title": f"{rouge_type.upper()} Scores Heatmap{suffix}",
                "label": f"{rouge_type.upper()} Score",
            }
        )

    # Save metrics to CSV
    with open(f"output/{filestem}_metrics.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "File1", "File2", "Value"])
        for metric in metric_configs:
            for data in metrics_data:
                value = (
                    data["rouge"][metric["name"]]
                    if "rouge" in metric["name"]
                    else data[metric["name"]]
                )
                writer.writerow([metric["name"], data["files"][0], data["files"][1], value])

    # Create heatmaps
    labels = [
        "_".join(file.replace(".md", "").split("_")[-2 if "textract" not in file else -1 :])
        for file in files
    ]
    for metric in metric_configs:
        ratios = [
            {
                "file1": data["files"][0],
                "file2": data["files"][1],
                "value": (
                    data["rouge"][metric["name"]]
                    if "rouge" in metric["name"]
                    else data[metric["name"]]
                ),
            }
            for data in metrics_data
        ]
        create_word_overlap_heatmap(
            files,
            ratios,
            labels,
            output_file=f"output/{metric['name']}_heatmap{suffix}.png",
            title=metric["title"],
            label=metric["label"],
            cmap="RdYlBu_r",
        )


def main() -> None:
    """Main function to evaluate similarity between markdown files given a filestem and a list
    of parsers."""
    parser = argparse.ArgumentParser(
        description="Compare the markdown files for a given input file and set of parsers."
    )
    parser.add_argument("-f", "--filestem", type=str, help="The base name of the files to compare")
    parser.add_argument(
        "-p",
        "--parser",
        type=str,
        nargs="+",
        default=["anthropic", "googleai", "mistral", "openai", "unstructuredio", "textract"],
        help="List of space-separated parsers to take into account",
    )
    parser.add_argument(
        "-m",
        "--markdown",
        action="store_true",
        help="Whether to remove markdown formatting from the files",
    )
    args = parser.parse_args()

    evaluate_all_files(args.filestem, args.parser, args.markdown)


if __name__ == "__main__":
    main()
