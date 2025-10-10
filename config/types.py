from pydantic import BaseModel
from typing import Optional, Literal


class InstrumentRegistry:
    def __init__(self, instruments):
        self.instruments = instruments
        self._label_to_index = {inst["label"]: inst["index"] for inst in instruments}
        self._name_to_index  = {inst["name"]: inst["index"] for inst in instruments}

    def __getitem__(self, index):
        """Cho phép truy cập bằng index, label hoặc name"""
        return self.instruments[index]

    def labels(self):
        return [inst["label"] for inst in self.instruments]

    def names(self):
        return [inst["name"] for inst in self.instruments]

    def indexes(self):
        return [inst["index"] for inst in self.instruments]

    def label_to_name(self, label):
        return self.instruments[self._label_to_index[label]]["name"]

    def label_to_index(self, label):
        return self._label_to_index[label]

    def name_to_label(self, name):
        idx = self._name_to_index[name]
        return self.instruments[idx]["label"]

    def index_to_label(self, index):
        return self.instruments[index]["label"]

    def index_to_name(self, index):
        return self.instruments[index]["name"]
    

class MelConfig(BaseModel):
    target_dbfs: float
    sr: int | float
    duration: float
    hop_length: int
    n_fft: int
    n_mels: int
    fmin: Optional[int]
    fmax: Optional[int]       # None -> sr/2
    top_db: int
    pre_emphasis: Optional[float]
    normalize: Literal["minmax_0_1", "zscore"] # "minmax_0_1" | "zscore"
    pad_mode: str