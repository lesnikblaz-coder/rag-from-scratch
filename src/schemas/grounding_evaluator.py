from pydantic import BaseModel, Field

from src.enums import GroundingLabel


class GroundingEvaluation(BaseModel):
    label: GroundingLabel  # evaluators classification
    reasoning: str = Field(
        description=(
            "Brief explanation of whether the evidence supports, "
            "contradicts, or fails to establish  the claim."
        )
    )

    """
    example:
    {
        'label': 'contradicted',
        'reasoning': 'The evidence specifies 25 working days, but the claim states 30.'
    }
    """