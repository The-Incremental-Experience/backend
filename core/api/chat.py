from ninja import Router
from django.http import HttpRequest

from api.schema.chat import Prediction

router = Router()


@router.post("/answer", response=Prediction)
def answer(request: HttpRequest):
    return Prediction(
        text="Hello, world!",
        source=["core.api.chat.answer"],
    )
