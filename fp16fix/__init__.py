"""fp16-fixer: find where a model's numbers leave float16's range, plan exact fixes, apply them at run time."""
from .find import Recorder, FP16_MAX
from .plan import Plan, save_fixes, load_fixes
from .apply import Applied
