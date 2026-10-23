from src.enums import GroundingLabel


GROUNDING_CASES = [
    {
        "id": "case_1",
        "evidence": [
            {
                "citation_id": "S1",
                "content": "Employees receive 25 working days of annual leave."
            }
        ],
        "claim": "Employees receive 25 working days of annual leave.", # A claim fully supported by the evidence.
        "expected_label": GroundingLabel.SUPPORTED
    },
    {
        "id": "case_2",
        "evidence": [
            {
                "citation_id": "S1",
                "content": "Employees receive 25 working days of annual leave."

            }
        ],
        "claim": "Employees receive 30 working days of annual leave.", # A claim directly contradicted by the evidence.
        "expected_label": GroundingLabel.CONTRADICTED
    },
    {
        "id": "case_3",
        "evidence": [
            {
                "citation_id": "S1",
                "content": "Employees receive 25 working days of annual leave."
            }
        ],
        "claim": "CEO's favorite food is pizza.", # A claim about information absent from the evidence.
        "expected_label": GroundingLabel.INSUFFICIENT_EVIDENCE
    },
    {
        "id": "case_4",
        "evidence": [
            {
                "citation_id": "S1",
                "content": "Employees receive 25 working days of annual leave."
            }
        ],
        "claim": "Employees are entitled to five weeks of annual vacation, based on a five-day working week.", # A claim that paraphrases the evidence correctly without using identical wording.
        "expected_label": GroundingLabel.SUPPORTED
    },
    {
        "id": "case_5",
        "evidence": [
            {
                "citation_id": "S1",
                "content": "Employees receive 25 working days of annual leave."
            }
        ],
        "claim": "Employees receive 25 vacation days, and all requests are automatically approved.", # A claim that combines two facts, where only one is supported.
        "expected_label": GroundingLabel.INSUFFICIENT_EVIDENCE
    }
]