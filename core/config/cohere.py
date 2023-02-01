from pydantic import BaseSettings


class CohereSettings(BaseSettings):
    key: str
    model: str
    max_tokens: int = 200
    temperature: int = 1

    class Config:
        env_prefix = "cohere_"
        case_sensitive = False
