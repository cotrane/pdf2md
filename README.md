# PDF to Markdown Converter

A Python tool that converts PDF files to Markdown format using various AI models. The tool supports multiple AI providers and models.

## Features

- Support for multiple AI providers:
  - Anthropic (Claude)
  - Google (Gemini)
  - OpenAI
  - Mistral
  - Ollama
  - Unstructured.io
- Configurable model selection for each provider
- Structured output with proper markdown formatting
- Environment variable configuration for API keys

## Installation

1. Clone the repository:

```bash
git clone https://github.com/cotrane/pdf2md.git
cd pdf2md
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
uv pip install -e ".[dev]"
```

## Configuration

### Environment Variables

Create a `.env` file similar to the `.env.tmpl` file and add your API keys.

The supported API keys are:

- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `UNSTRUCTURED_API_KEY`

In order to test AWS Textract you require an AWS account and credentials which can then be added to the `.env` file using

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

## Usage

### Basic Usage

```bash
uv run src/run.py --input input.pdf --parser anthropic
```

The output file will be created in folder `output` and be called `<input_file_name>_<parser>_<model>.md`.

### Available Parsers

- `anthropic`: Uses Claude 3.5 Sonnet
- `googleai`: Uses Google's Gemini Pro
- `openai`: Uses GPT-4 Turbo
- `mistral`: Uses Mistral Large
- `ollama`: Uses Ollama models
- `textract`: Uses Textract for text extraction
- `unstructuredio`: Uses Unstructured.io API

### Model Selection

Each parser supports different models. Use the `--model` option to specify a model:

```bash
uv run src/run.py --input input.pdf --parser anthropic --model claude-3-sonnet-20240229
```

Available models per parser:

- Anthropic:
  - `claude-3-7-sonnet-20250219` (default)
  - `claude-3-5-sonnet-20241022`
- Google AI:
  - `gemini-1.5-flash`
  - `gemini-2.0-flash` (default)
  - `gemini-2.0-flash-thinking-exp-01-21`
  - `gemini-2.5-pro-exp-03-25`
- OpenAI:
  - `gpt-4o` (default)
  - `gpt-4o-mini`
  - `gpt-4.5-preview`
  - `o1`
- Mistral: `mistral-ocr-latest`
- Ollama: Any model available in your Ollama installation
- Textract: No model selection
- Unstructured.io:
  - `gpt-4o` (default)
  - `claude-3-5-sonnet-20241022`
  - `gemini-2.0-flash-001`
  - `hi-res`
  - `fast`

### Evaluating Results

The tool includes a utility to evaluate the similarity between markdown files. This script creates a similarity heatmap between the various output files and serves as a rough measure of how accurate the created markdown file is once one is checked manually.

```bash
uv run src/evaluate.py -f <filename>
```

The evaluation provides several metrics:

- Cosine Similarity: TF-IDF based similarity score (0-1)
- Word Overlap Ratio: Ratio of common words to total unique words (0-1)
- Levenshtein Ratio: Normalized similarity score based on Levenshtein distance (0-1)
- Word Statistics: Counts of words, common words, and unique words in each file

In order to evaluate only the OCR part of the output file, we can remove all markdown notation by running the script as follows:

```bash
uv run src/evaluate.py -f <filename> -m
```

## Running Tests

### Unit Tests

To run all unit tests:
```bash
uv run pytest tests/ -v
```

To run tests with coverage:
```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

### Integration Tests

Integration tests require external services to be configured. To run them:
```bash
uv run pytest tests/ -v -m integration
```

### Test Categories

- `test_base.py`: Tests for the base parser functionality
- `test_run.py`: Tests for the main script functionality
- `test_integration.py`: Integration tests requiring external services

## Test Configuration

The test suite is configured in `pyproject.toml` with the following settings:

- Test paths: `tests/`
- Test file pattern: `test_*.py`
- Coverage reporting: Enabled
- Integration test marker: `@pytest.mark.integration`

## Development

### Code Style

The project uses:

- Pylint for code linting (configured in pyproject.toml)
- Black for code formatting
- MyPy for type checking

### Development Dependencies

Install development dependencies:

```bash
uv pip install ".[dev]"
```

### Evaluation Dependencies

Install evaluation dependencies:

```bash
uv pip install ".[eval]"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
