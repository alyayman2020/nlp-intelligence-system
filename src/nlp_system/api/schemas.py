"""Pydantic schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, examples=["This product exceeded my expectations!"]
    )
    dataset: str = Field("amazon_fine_food", description="Which trained model to use.")


class PredictResponse(BaseModel):
    label: int = Field(..., description="0 = negative, 1 = positive")
    sentiment: str
    confidence: float
    dataset: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["battery life disappointing"])
    top_k: int = Field(10, ge=1, le=100)
    sentiment: int | None = Field(
        None, description="Filter: 0 negative, 1 positive, null = no filter."
    )
    dataset: str = Field("amazon_fine_food")


class SearchHitOut(BaseModel):
    doc_id: int
    score: float
    text: str
    sentiment: int


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchHitOut]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    indexes_loaded: list[str]
