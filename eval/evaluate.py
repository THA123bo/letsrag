"""RAG quality evaluation with DeepEval (Faithfulness + Relevancy).

EVALUATION METRICS OVERVIEW:
1. Faithfulness (FaithfulnessMetric): 
   - WHAT IT MEASURES: Evaluates whether the LLM's final generated answer is factually grounded in the retrieved context. 
   - HOW IT MEASURES: The LLM Judge extracts all statements from the actual_output and systematically checks if each statement can be logically deduced from the retrieval_context. If a statement contradicts the context or invents new information (hallucinations), the score drops.

2. Answer Relevancy (AnswerRelevancyMetric):
   - WHAT IT MEASURES: Evaluates how directly the LLM answered the user's original question, penalizing rambling, evasive, or incomplete answers.
   - HOW IT MEASURES: The LLM Judge generates hypothetical questions based on the actual_output and compares them against the original user input. High similarity means the answer was highly relevant.

3. Contextual Recall (ContextualRecallMetric):
   - WHAT IT MEASURES: Evaluates the RETRIEVAL QUALITY — whether the retrieved context chunks contain all the information needed to answer the question correctly.
   - HOW IT MEASURES: The LLM Judge checks if each key claim in the expected_output can be traced back to the retrieval_context. A low score means the search engine (BM25+Semantic+Reranker) is not surfacing the right documents.

NOTE ON LOCAL LLM EVALUATION:
Because we are using a local LLM (e.g., llama3.1:8b) as the judge, we have disabled strict formatting (strict_mode=False) and textual reasoning generation (include_reason=False). Forcing a local LLM to generate complex JSON reasoning often drains its context capacity and leads to false zero scores. By allowing it to compute only the numeric score, we achieve a much more accurate evaluation.
"""

import json
import sys
import os

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from rag_studio.config import Config
from rag_studio.llm import llm_chat
from rag_studio.retrieval import search_hybrid
from rag_studio.prompts import SYSTEM_PROMPT, build_user_prompt

DATASET_PATH = Config.EVAL_DIR / "dataset.json"
RESULTS_PATH = Config.EVAL_DIR / "results.json"


class CustomRAGLLM(DeepEvalBaseLLM):
    """Wrapper to use our llm_chat() with DeepEval.
    
    Attributes:
        model_name (str): The name of the LLM model to use.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        """Loads and returns the model name."""
        return self.model_name

    def generate(self, prompt: str) -> str:
        """Generates a response for DeepEval, using JSON mode when structured output is needed.

        DeepEval's evaluation prompts always contain specific JSON schema instructions.
        We detect this and force Ollama's format='json' mode to get guaranteed valid JSON,
        avoiding the JSONDecodeError that occurs when models wrap output in markdown or prose.
        """
        import ollama as _ollama
        messages = [{"role": "user", "content": prompt}]
        
        # DeepEval evaluation prompts always ask for JSON with specific schemas.
        # Use Ollama's native JSON mode to guarantee valid output.
        response = _ollama.chat(
            model=self.model_name,
            messages=messages,
            format="json"
        )
        return response["message"]["content"].strip()

    async def a_generate(self, prompt: str) -> str:
        """Async version: falls back to synchronous generate with JSON mode."""
        return self.generate(prompt)

    def get_model_name(self):
        """Returns the model name."""
        return self.model_name


def load_dataset(file_path: str) -> list[dict]:
    """Loads a JSON dataset from the given file path.

    Args:
        file_path (str): The path to the JSON evaluation dataset.

    Returns:
        list[dict]: A list of dictionary test cases.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    """Executes the evaluation pipeline over all questions in the dataset."""
    dataset = load_dataset(DATASET_PATH)
    evaluator = CustomRAGLLM(model_name=Config.LLM_MODEL)

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.5, 
        model=evaluator, 
        strict_mode=False, 
        include_reason=False
    )
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5, 
        model=evaluator, 
        strict_mode=False, 
        include_reason=False
    )
    recall_metric = ContextualRecallMetric(
        threshold=0.5,
        model=evaluator,
        strict_mode=False,
        include_reason=False
    )

    results = []

    for i, item in enumerate(dataset):
        question = item["question"]
        print(f"[{i+1}/{len(dataset)}] Evaluating: {question}...", flush=True)
        expected_output = item.get("expected_output") or item.get("answer")

        retrieved_chunks = search_hybrid(question, limit=Config.RETRIEVAL_LIMIT)
        retrieval_context = [chunk["text"] for chunk in retrieved_chunks]

        user_prompt = build_user_prompt(retrieval_context, question)
        actual_output = llm_chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        try:
            faithfulness_metric.measure(test_case)
            faith_score = faithfulness_metric.score
            faith_reason = faithfulness_metric.reason
            faith_passed = faithfulness_metric.is_successful()
        except Exception as e:
            print(f"  [WARN] Faithfulness evaluation failed: {e}", flush=True)
            faith_score = None
            faith_reason = f"Evaluation error: {type(e).__name__}"
            faith_passed = False

        try:
            relevancy_metric.measure(test_case)
            rel_score = relevancy_metric.score
            rel_reason = relevancy_metric.reason
            rel_passed = relevancy_metric.is_successful()
        except Exception as e:
            print(f"  [WARN] Relevancy evaluation failed: {e}", flush=True)
            rel_score = None
            rel_reason = f"Evaluation error: {type(e).__name__}"
            rel_passed = False

        try:
            recall_metric.measure(test_case)
            recall_score = recall_metric.score
            recall_reason = recall_metric.reason
            recall_passed = recall_metric.is_successful()
        except Exception as e:
            print(f"  [WARN] Recall evaluation failed: {e}", flush=True)
            recall_score = None
            recall_reason = f"Evaluation error: {type(e).__name__}"
            recall_passed = False

        evaluation_data = {
            "question": question,
            "expected_output": expected_output,
            "actual_output": actual_output,
            "retrieval_context": retrieval_context,
            "metrics": [
                {
                    "name": "Faithfulness",
                    "score": round(faith_score, 2) if faith_score is not None else None,
                    "reason": faith_reason,
                    "passed": faith_passed,
                },
                {
                    "name": "Answer Relevancy",
                    "score": round(rel_score, 2) if rel_score is not None else None,
                    "reason": rel_reason,
                    "passed": rel_passed,
                },
                {
                    "name": "Contextual Recall",
                    "score": round(recall_score, 2) if recall_score is not None else None,
                    "reason": recall_reason,
                    "passed": recall_passed,
                },
            ],
        }
        results.append(evaluation_data)

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        faith_display = f"{faith_score:.2f}" if faith_score is not None else "ERR"
        rel_display = f"{rel_score:.2f}" if rel_score is not None else "ERR"
        recall_display = f"{recall_score:.2f}" if recall_score is not None else "ERR"
        print(
            f"  Faith: {faith_display} | Rel: {rel_display} | Recall: {recall_display} (saved to {RESULTS_PATH})",
            flush=True,
        )


if __name__ == "__main__":
    run_evaluation()
