import json
from typing import List, Dict

import cohere
import annoy
import numpy
from api.schema.chat import Prediction, Question
from config.config import config
from django.http import HttpRequest
from ninja import Router

router = Router()


@router.post("/answer", response=Prediction)
async def answer(_: HttpRequest, question: Question):
    co = cohere.Client(config.cohere.key)
    response_text = generate_response(co, question.text)
    sources = find_sources(co, question.text)

    return Prediction(
        text=response_text,
        sources=sources,
    )


def generate_response(co, question_text) -> str:
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

    response = co.generate(
        prompt=question_text,
        model=config.cohere.model,
        **kwargs,
    )

    response_text = ".".join(response[0].text.split(".")[:-1]) + "."
    return response_text


def find_sources(co, question_text) -> List[str]:
    JSON_IN = "core/data/question_to_source_map.json"
    INDEX_IN = "core/data/index.ann"
    INDEX_SHAPE = 4096
    # the only one that works for embeddings
    MODEL = "large"

    ordered_question_to_source_map: List[Dict[str, str]] = json.load(open(JSON_IN, "r"))

    source_index = annoy.AnnoyIndex(INDEX_SHAPE, "angular")
    source_index.load(INDEX_IN)

    question_embedding = numpy.array(co.embed(texts=[question_text], model=MODEL).embeddings)[0]

    similar_item_ids = source_index.get_nns_by_vector(question_embedding, 10)

    # return best 3 matches
    sources = [ordered_question_to_source_map[similar_item_ids[i]]["source"] for i in range(len(similar_item_ids[:3]))]
    return list(set(sources))
