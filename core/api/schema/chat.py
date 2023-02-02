"""Schemas for the Psy-Q bot API.
"""

from typing import List
from ninja import Schema


class Prediction(Schema):
    """Model prediction with the list of sources."""

    text: str
    sources: List[str]


class Question(Schema):
    """Cohere prompt"""

    text: str