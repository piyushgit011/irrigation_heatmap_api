from pydantic import BaseModel

class Irrigation(BaseModel):
    array: list[list[int]]