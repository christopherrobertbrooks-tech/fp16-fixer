"""fp16-fixer: find where a model's numbers leave float16's range, and (later) plan exact fixes."""
from .find import Recorder, FP16_MAX
