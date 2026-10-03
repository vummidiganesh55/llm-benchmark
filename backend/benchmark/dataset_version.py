import hashlib
import json
from pathlib import Path
from typing import Any


class DatasetVersion:

    def __init__(
        self,
        dataset_path: str = "datasets/benchmark.json",
        metadata_path: str = "datasets/metadata.json"
    ):
        self.dataset_path = Path(dataset_path)
        self.metadata_path = Path(metadata_path)

    def calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of the benchmark dataset.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        data = self.dataset_path.read_bytes()

        return hashlib.sha256(data).hexdigest()

    def load_metadata(self) -> dict[str, Any]:
        """
        Load dataset metadata.
        """

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def get_info(self) -> dict[str, Any]:
        """
        Return complete dataset version information.
        """

        metadata = self.load_metadata()

        return {
            "name": metadata.get("name"),
            "version": metadata.get("version"),
            "description": metadata.get("description"),
            "file": metadata.get("file"),
            "question_count": metadata.get("question_count"),
            "categories": metadata.get("categories"),
            "sha256": self.calculate_hash()
        }