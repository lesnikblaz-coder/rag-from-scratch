import json

from google import genai

from src.schemas.grounding_evaluator import GroundingEvaluation
from src.config import settings
from src.evaluation.prompts import GROUNDING_EVALUATION_INSTRUCTION


class GroundedEvaluator:
    def __init__(self, client: genai.client.AsyncClient):
        self.client = client

    async def evaluate(self, *, claim: str, evidence: list[dict]) -> GroundingEvaluation:
        prompt = (
            f"Claim:\n{claim}\n\n"
            f"Evidence (JSON):\n"
            f"{json.dumps(evidence, ensure_ascii=False)}"
        )

        interaction = await self.client.interactions.create(
            model=settings.INTERACTION_MODEL,
            system_instruction=GROUNDING_EVALUATION_INSTRUCTION,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": GroundingEvaluation.model_json_schema()
            }
        )

        return GroundingEvaluation.model_validate_json(
            interaction.output_text
        )