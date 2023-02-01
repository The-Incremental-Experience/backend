from config.cohere import CohereSettings


class BackendConfig:
    cohere: CohereSettings = CohereSettings()


config = BackendConfig()
