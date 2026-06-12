from pydantic import BaseModel, HttpUrl

class VideoProcessRequest(BaseModel):
    url: HttpUrl