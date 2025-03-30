#!/usr/bin/env python3
"""Script to evaluate similarity between markdown files given a filestem and a list of parsers."""

import argparse
import glob
import itertools
import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Levenshtein import distance, ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logging import setup_logging

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

    # Remove bold and italic
    text = re.sub(r"[*_~`]", "", text)

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

    # Remove tables
    text = re.sub(r"^\s*\|.*\|$", "", text, flags=re.MULTILINE)  # Table rows
    text = re.sub(r"^\s*\|[-:|\s]+\|\s*$", "", text, flags=re.MULTILINE)  # Table separators

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
        r"\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\zeta|\\eta|\\theta|\\iota|\\kappa|\\lambda|\\mu|\\nu|\\xi|\\pi|\\rho|\\sigma|\\tau|\\upsilon|\\phi|\\chi|\\psi|\\omega",
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
    file_to_idx = {file: idx for idx, file in enumerate(files)}

    # Fill the matrix with overlap ratios
    for ratio_data in overlap_ratios:
        i = file_to_idx[str(ratio_data["file1"])]
        j = file_to_idx[str(ratio_data["file2"])]
        matrix[i, j] = float(ratio_data["value"])
        matrix[j, i] = float(ratio_data["value"])  # Make the matrix symmetric

    # Create heatmap
    plt.figure(figsize=(12, 12))  # Increased figure size for better readability
    ax = sns.heatmap(
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
    logger.info(f"Found {len(files)} files for filestem {filestem} and parsers {parsers}")
    logger.info(files)
    return files


def evaluate_files(file1: str, file2: str, preprocess: bool = True) -> tuple[float, float, float]:
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
        Tuple containing (cosine similarity, word overlap ratio, levenshtein ratio).
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
    levenshtein_dist, levenshtein_ratio = calculate_levenshtein_metrics(clean_text1, clean_text2)

    # logger.info results
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)
    logger.info(f"Cosine Similarity: {cosine_sim:.4f}")
    logger.info(f"Word Overlap Ratio: {overlap_ratio:.4f}")
    logger.info(f"Levenshtein Distance: {levenshtein_dist}")
    logger.info(f"Levenshtein Ratio: {levenshtein_ratio:.4f}")
    logger.info(f"\nWords in first file: {len(words1)}")
    logger.info(f"Words in second file: {len(words2)}")
    logger.info(f"Common words: {len(words1.intersection(words2))}")
    logger.info(f"Unique to first file: {len(words1 - words2)}")
    logger.info(f"Unique to second file: {len(words2 - words1)}")

    # logger.info unique words
    logger.info("\nUnique words in first file:")
    logger.info("-" * 50)
    logger.info(sorted(words1 - words2))
    logger.info("\nUnique words in second file:")
    logger.info("-" * 50)
    logger.info(sorted(words2 - words1))

    return cosine_sim, overlap_ratio, levenshtein_ratio


def evaluate_all_files(filestem: str, parsers: list[str], remove_markdown: bool = True) -> None:
    """Evaluate similarity between all pairs of markdown files.

    Args:
        filestem: The base name of the files to compare.
        parsers: List of space-separated parsers to take into account.
        remove_markdown: Whether to remove markdown formatting from the files.
            Default is True.
    """
    # Find all files with the given filestem and parsers
    files = find_files(filestem, parsers)

    # Calculate overlap ratios for all pairs of files
    overlap_ratios, levenshtein_ratios, cosine_ratios = [], [], []
    for file1, file2 in itertools.combinations_with_replacement(files, 2):
        cosine_sim, overlap_ratio, levenshtein_ratio = evaluate_files(
            file1, file2, preprocess=remove_markdown
        )
        overlap_ratios.append({"file1": file1, "file2": file2, "value": overlap_ratio})
        levenshtein_ratios.append({"file1": file1, "file2": file2, "value": levenshtein_ratio})
        cosine_ratios.append({"file1": file1, "file2": file2, "value": cosine_sim})

    # Create and save heatmap
    suffix = " with Markdown" if not remove_markdown else ""
    metrics = [
        {
            "ratios": overlap_ratios,
            "name": "word_overlap",
            "title": f"Word Overlap Ratios Heatmap{suffix}",
            "label": "Overlap Ratio",
        },
        {
            "ratios": levenshtein_ratios,
            "name": "levenshtein",
            "title": f"Levenshtein Ratios Heatmap{suffix}",
            "label": "Levenshtein Ratio",
        },
        {
            "ratios": cosine_ratios,
            "name": "cosine",
            "title": f"Cosine Ratios Heatmap{suffix}",
            "label": "Cosine Ratio",
        },
    ]

    suffix = "_w_markdown" if not remove_markdown else ""
    # Extract parser names from filenames (e.g., "output/test_anthropic.md" -> "anthropic")
    labels = [
        "_".join(file.replace(".md", "").split("_")[-2 if not "textract" in file else -1 :])
        for file in files
    ]
    for metric in metrics:
        create_word_overlap_heatmap(
            files,
            metric["ratios"],
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
