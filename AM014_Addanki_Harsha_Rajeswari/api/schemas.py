from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^[0-9A-Za-z_-]{1,64}$", examples=["00000056"])
