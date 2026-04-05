"""
CFPB Consumer Complaint Loader.

Reads the CFPB JSON file downloaded by scripts/download_datasets.py
and returns a list of document dicts ready for chunking.

Each document dict:
    {
        "text": str,          # consumer_complaint_narrative
        "source": str,        # "cfpb_complaint"
        "metadata": {
            "complaint_id": str,
            "product": str,
            "sub_product": str,
            "issue": str,
            "state": str,
            "date_received": str,
            "source_type": "cfpb_complaint",
            "eval_only": False,
        }
    }
"""

import json
from pathlib import Path
from typing import Any


class CFPBComplaintLoader:
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> list[dict[str, Any]]:
        """Load complaints and return list of document dicts."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CFPB file not found: {self.file_path}")

        with open(self.file_path, encoding="utf-8") as f:
            raw = json.load(f)

        documents = []
        for hit in raw:
            source = hit.get("_source", {})
            narrative = source.get("consumer_complaint_narrative", "").strip()

            # Skip empty narratives
            if not narrative or len(narrative) < 50:
                continue

            documents.append(
                {
                    "text": narrative,
                    "source": str(self.file_path),
                    "metadata": {
                        "complaint_id": str(source.get("complaint_id", "")),
                        "product": source.get("product", ""),
                        "sub_product": source.get("sub_product", ""),
                        "issue": source.get("issue", ""),
                        "state": source.get("state", ""),
                        "date_received": source.get("date_received", ""),
                        "source_type": "cfpb_complaint",
                        "eval_only": False,
                    },
                }
            )

        return documents
