import re

from src.exceptions import custom as err


def validate_citations(answer: str, evidence_registry: dict) -> list[str]:
    reg_cits = evidence_registry.keys()
    answer_cits = re.findall(r'\b(S\d+)\b', answer)

    if not answer_cits:
        raise err.InvalidCitationError()

    for c in answer_cits:
        if c not in reg_cits:
            raise err.InvalidCitationError()

            #list(dict.fromkeys()) preserves the order of the first occurrence
    return list(dict.fromkeys(answer_cits))