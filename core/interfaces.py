from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Variant:
    name: str          # "control" or "treatment"
    config: dict = field(default_factory=dict)  # arm-specific config (corpus path, etc.)


class Task(ABC):
    """Given a query and a variant, produce an output dict.

    AEO implementation: retrieve top-k chunks from the variant's corpus,
    call the LLM, return answer text + retrieved chunk IDs.
    """

    @abstractmethod
    def run(self, query: str, variant: Variant) -> dict:
        ...


class Evaluator(ABC):
    """Given a query and a task output, return structured scores.

    AEO implementation: LLM-as-judge checks whether the answer recommends
    the target product and assigns a simulated acceptance signal.
    """

    @abstractmethod
    def evaluate(self, query: str, output: dict, variant: Variant) -> dict:
        ...


class Metric(ABC):
    """Reduce a list of per-trial score dicts to a single number per arm.

    AEO implementation: recommendation rate = mean(recommended) per arm.
    """

    @abstractmethod
    def compute(self, results: list[dict]) -> dict:
        ...
