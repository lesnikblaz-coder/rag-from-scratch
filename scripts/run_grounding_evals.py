import asyncio

from google import genai

from evals.grounding_cases import GROUNDING_CASES
from src.config import settings
from src.evaluation.grounding_evaluator import GroundedEvaluator


async def main():
    aclient = genai.Client(
        api_key=settings.GEMINI_API_KEY
    ).aio

    try:
        evaluator = GroundedEvaluator(aclient)

        correct = 0

        for gc in GROUNDING_CASES:
            evaluation = await evaluator.evaluate(
                claim=gc["claim"],
                evidence=gc["evidence"]
            )

            hit = evaluation.label == gc["expected_label"]

            if hit:
                correct += 1

            print(gc["id"])
            print("Claim: ", gc['claim'])
            print("Expected: ", gc['expected_label'])
            print("Predicted: ", evaluation.label)
            print("Reasoning: ", evaluation.reasoning)
            print("Result: ", "PASS" if hit else "FAIL", "\n")


        accuracy = (correct / len(GROUNDING_CASES)) * 100

        print("\n================================")
        print(f"TOTAL CASES: {len(GROUNDING_CASES)}")
        print(f"CORRECT: {correct}")
        print(f"ACCURACY: {accuracy}%")
        print("================================")

    finally:
        await aclient.aclose()

if __name__ == "__main__":
    asyncio.run(main())