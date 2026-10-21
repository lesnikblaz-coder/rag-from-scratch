import re

from src.exceptions import custom as err


def validate_citations(answer: str, evidence_registry: dict) -> list[str]:
    reg_cits = evidence_registry.keys()
    answer_cits = list(set(re.findall(r'\bS\d+\b', answer)))

    for c in answer_cits:
        if c not in reg_cits:
            raise err.InvalidCitationError()

    return answer_cits