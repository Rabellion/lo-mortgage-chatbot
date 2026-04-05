from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "lo-mortgage-chatbot"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAFT synthesis
    openai_api_key: str = ""
    synthesis_model: str = "gpt-4o"
    synthesis_concurrency: int = 30
    raft_target_triplets: int = 20000

    # Fine-tuning
    hf_token: str = ""
    base_model: str = "google/gemma-4-31b-it"
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    num_epochs: int = 3

    # Experiment tracking
    wandb_api_key: str = ""
    wandb_project: str = "lo-mortgage-chatbot"

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [self.data_raw_dir, self.data_processed_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
