"""Ollama parser for PDF to Markdown conversion."""

import base64
from typing import List

from ollama import ChatResponse, Client
from pdf2image import convert_from_path
from PIL import Image

from .base import BaseParser


class OllamaParser(BaseParser):
    """A class to handle PDF to Markdown conversion using Ollama models."""

    AVAILABLE_MODELS = ["gemma3:4b", "gemma3:12b", "llama3.2-vision:11b"]
    DEFAULT_MODEL = "gemma3:4b"
    MAX_IMAGE_WIDTH = 1920

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
        base_url: str = "http://localhost:11434",
    ) -> None:
        """Initialize the Ollama parser.

        Args:
            model: The name of the Ollama model to use.
            temperature: The temperature parameter for text generation.
            top_p: The top_p parameter for text generation.
            top_k: The top_k parameter for text generation.
            max_tokens: The maximum number of tokens to generate.
            base_url: The base URL for the Ollama API. Defaults to http://localhost:11434.

        Raises:
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model, temperature, top_p, top_k, max_tokens)
        self.base_url = base_url
        self.client = Client(host=base_url)
        self.model_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": max_tokens,
        }
        self.logger.debug("Ollama parser initialization complete")

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to maintain aspect ratio with max width.

        Args:
            image: PIL Image to resize.

        Returns:
            Resized PIL Image.
        """
        if image.width > self.MAX_IMAGE_WIDTH:
            ratio = self.MAX_IMAGE_WIDTH / image.width
            new_height = int(image.height * ratio)
            return image.resize((self.MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
        return image

    def read_pdf_as_base64_img(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to base64-encoded images.

        Args:
            pdf_path: Path to the PDF file.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.

        Returns:
            List[str]: List of base64-encoded image strings.
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)

            # Convert each image to base64
            base64_images = []
            for image in images:
                # Resize image if needed
                image = self.resize_image(image)

                import io

                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                base64_images.append(base64.b64encode(img_byte_arr).decode("utf-8"))

            return base64_images
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}", exc_info=True)
            raise

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF file to markdown using Ollama.

        Args:
            pdf_path: The path to the PDF file to convert.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.

        Returns:
            str: The converted markdown text.
        """
        self.logger.debug(f"Starting PDF to Markdown conversion for file: {pdf_path}")

        try:
            # Convert PDF to base64 images
            self.logger.debug(f"Converting PDF to images: {pdf_path}")
            base64_images = self.read_pdf_as_base64_img(pdf_path)

            prompt = "Convert the attached pdf to markdown format for me please."

            # Prepare the request payload
            response: ChatResponse = self.client.chat(
                model=self.model,
                options=self.model_config,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": base64_images,
                    },
                ],
            )

            self.logger.debug("Successfully received response from Ollama API")
            if not response.message.content:  # pylint: disable=all
                raise ValueError("Empty response from Ollama API")
            return response.message.content  # pylint: disable=all

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
