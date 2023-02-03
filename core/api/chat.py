import cohere
from api.schema.chat import Prediction, Question
from config.config import config
from django.http import HttpRequest
from ninja import Router

router = Router()


@router.post("/answer", response=Prediction)
async def answer(_: HttpRequest, question: Question):
    kwargs = dict(
        max_tokens=config.cohere.max_tokens,
        temperature=config.cohere.temperature,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods="NONE",
    )

    co = cohere.Client(config.cohere.key)
    response = co.generate(
        prompt=question.text,
        model=config.cohere.model,
        **kwargs,
    )

    return Prediction(
        text=response[0].text,
        sources=["Book 1 Page 3"],
    )
