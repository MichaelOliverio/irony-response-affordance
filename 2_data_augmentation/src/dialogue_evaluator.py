"""
Automatic evaluation of generated dialogues using perplexity and LLM-as-a-judge.
"""
import re
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from pathlib import Path
from evaluate import load

from .llm_handler import LLMHandler
from .config import DATA_AUGMENTATION_OUTPUT_DIR

class DialogueEvaluator:
    """Class for automatic evaluation of generated dialogues."""
    
    # Modelli per perplexity evaluation
    PERPLEXITY_MODELS = [
        "meta-llama/Llama-3.2-1B-Instruct"
    ]
    
    # Modelli per LLM-as-a-judge
    JUDGE_MODELS = [
        "meta-llama/Llama-3.2-1B-Instruct",
    ]

    # aggiornare
    JUDGE_DIMENSIONS = {
        'Coherence': {
            'definition': 'the quality of a generated turn that allows it to fit logically and meaningfully within the following context',
            'questions': {
                'Logical Flow': 'Does the generated turn logically precede the original turn, maintaining a clear flow of ideas?',
                'Relevance': 'Is the generated turn directly relevant to the content and context of the dialogue?',
                'Continuity 1': 'Does the generated turn stay consistent with the topic discussed in the dialogue?',
                'Continuity 2': 'Does the generated turn integrate smoothly with the ongoing conversation, ensuring a coherent progression?'
            }
        },
        'Engagingness': {
            'definition': 'the extent to which a generated turn captures and maintains a reader’s interest',
            'questions': {
                'Informative': 'Does the generated turn add meaningful value to the conversation?',
                'Emotional Engagement': 'Is the generated turn friendly, polite, and empathetic?',
                'Interest Level': 'Does the generated turn contribute to a more engaging and interesting conversation?',
            }
        },
        'Groundedness': {
            'definition': 'the accuracy and relevance of a generated turn in relation to the factual information it is based on',
            'questions': {
                'Relevance 1': 'Does the generated turn provide new information while maintaining the flow of the conversation?',
                'Relevance 2': 'Does it effectively utilize the key information that has been mentioned in the conversation?',
                'Consistency 1': 'Does the generated turn remain consistent with the original turn?',
                'Consistency 2': 'Does it avoid contradicting afterwards provided information?'
            }
        },
        'Naturalness': {
            'definition': 'the degree to which a generated turn resembles something a real person would naturally say',
            'questions': {
                'Avoid repetition': 'Does the generated turn avoid unnecessary repetition of the same content between sentences?',
                'Context relevance': 'Are the sentences used naturally within the context?',
                'Clarity': 'Is the overall message clear and easy to understand?',
                'Word choice and tone 1': 'Is the tone consistent throughout?',
                'Word choice and tone 2': 'Is the text free of major grammatical errors?'
            }
        }
    }
    
    def __init__(self, perplexity_models: Optional[List[str]] = None, judge_models: Optional[List[str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            perplexity_model: Model to use for perplexity calculation
        """
        if perplexity_models is None:
            raise ValueError("Please specify at least one perplexity model."
                             f" Supported models: {self.PERPLEXITY_MODELS}")
        
        # check if the models are in the supported list
        for model in perplexity_models:
            if model not in self.PERPLEXITY_MODELS:
                raise ValueError(f"Perplexity model '{model}' is not supported. Supported models: {self.PERPLEXITY_MODELS}")

        if judge_models is None:
            raise ValueError("Please specify at least one judge model."
                             f" Supported models: {self.JUDGE_MODELS}")
        
        for model in judge_models:
            if model not in self.JUDGE_MODELS:
                raise ValueError(f"Judge model '{model}' is not supported. Supported models: {self.JUDGE_MODELS}")

        self.perplexity_models = perplexity_models
        self.judge_models = judge_models
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def eval(
        self, 
        dataset_df: pd.DataFrame,
        output_file: Path,
        metrics: List[str] = ["perplexity", "judge"],
    ):
        """
        Evaluate all dialogues in a dataset.
        
        Args:
            dataset_df: DataFrame with dialogue data
            output_file: Path to save results
            compute_perplexity: Whether to compute perplexity
            compute_judge: Whether to use LLM judge
            judge_models: List of judge models to use (default: self.JUDGE_MODELS)
        """
        results = {}

        if "perplexity" in metrics:            
            perplexity_results = {}
            for model_name in self.perplexity_models:
                print(f"Calculating perplexity for model: {model_name}")
                perplexity_results[model_name] = []
                self.pipe, self.model, self.tokenizer = LLMHandler.load_model(model_name)

                for row in dataset_df.itertuples():
                    print(f"Calculating perplexity for dialogue ID: {row.unique_id}")
                    turns = row.dialogue['turns']
                    ppl_metrics = self._calculate_dialogue_perplexity(model_name, turns)

                    result = {
                        'unique_id': row.unique_id,
                        'dataset_name': row.dataset_name,
                        'dialogue': row.dialogue,
                        **ppl_metrics
                    }

                    perplexity_results[model_name].append(result)

                LLMHandler.destroy_model(self.model, self.tokenizer)
                print(f"✓ Perplexity calculation complete for model: {model_name}\n")

            results['perplexity'] = perplexity_results



        # vedi figura 6 -checkeval in github: prompt/topical_chat_questions trovo le definizioni
        # usiamo il loro prompt e lo modifichiamo un pochettino

        
        # LLM Judge evaluation
        if "llm-judge" in metrics:
            judge_results = {}
            for model_name in self.judge_models:      
                print(f"🤖 Evaluating with judge model: {model_name}")
                
                judge_results[model_name] = []
                self.pipe, self.model, self.tokenizer = LLMHandler.load_model(model_name)
                for row in dataset_df.itertuples():
                    print(f"Calculating judge score for dialogue ID: {row.unique_id}")
                    print(f" Dialogue: {row.dialogue}")
                    turns = row.dialogue['turns']
                    judge_metrics = self._generate_judge_score(model_name, turns)

                    print(f" Judge metrics: {judge_metrics}")

                    result = {
                        'unique_id': row.unique_id,
                        'dataset_name': row.dataset_name,
                        'dialogue': row.dialogue,
                        **judge_metrics
                    }

                    judge_results[model_name].append(result)

                LLMHandler.destroy_model(self.model, self.tokenizer)
                print(f"✓ Judge evaluation complete for model: {model_name}\n")

            results['judge'] = judge_results

        # Save results
        results_df = pd.DataFrame(results)
        
        # Save as JSON
        results_df.to_json(output_file, orient='records', indent=2, force_ascii=False)
        print(f"✅ Results saved to: {output_file}")
            
        return results_df

        
    # ========== Perplexity Methods ========== #
    def _calculate_perplexity(self, model_name: str, predictions: List[str]) -> float:
        """
        Calculate perplexity of a text using the loaded model.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score (lower is better)
        """
        try:
            perplexity = load("pico-lm/perplexity")
        except Exception as e:
            print(f"⚠ Error loading perplexity metric: {e}")
            return float('inf')

        results = perplexity.compute(predictions=predictions, model_id='gpt2')        
        return results
    
    def _calculate_dialogue_perplexity(self, model_name: str, turns: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Calculate perplexity metrics for a dialogue.
        
        Args:
            turns: List of turn dicts with 'u1' and 'u2' keys
            
        Returns:
            Dict with perplexity metrics
        """
        # Global dialogue perplexity
        lines = []
        for turn in turns:
            if 'u1' in turn:
                lines.append(f"u1: {turn['u1']}")
            if 'u2' in turn:
                lines.append(f"u2: {turn['u2']}")
        dialogue = "\n".join(lines)
        overall_ppl = self._calculate_perplexity(model_name, [dialogue])
        
        # Per-turn perplexity (excluding last turn which is original)
        utterances = []
        for i, turn in enumerate(turns[:-1]):  # Exclude last turn
            if turn.get('u1'):
                utterances.append(turn['u1'])
            if turn.get('u2'):
                utterances.append(turn['u2'])
        per_turn_ppls = self._calculate_perplexity(model_name, utterances)

        return {
            "overall_perplexity": overall_ppl['perplexities'],
            "per_turn_perplexities": per_turn_ppls['perplexities'],
            "per_turn_avg_perplexity": np.mean(per_turn_ppls['perplexities']) if per_turn_ppls else None,
        }
    
    # ========== LLM-as-a-Judge Methods ========== #
    def _generate_judge_score(self, model_name: str, turns: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Generate LLM judge scores for a dialogue across multiple dimensions.
        
        Args:
            model_name: Name of the judge model
            turns: List of turn dicts with 'u1' and 'u2' keys
            
        Returns:
            Dict with judge scores for each dimension
        """
        results = {}
        
        
        # Get the response to evaluate (the preceding turn before the last one)
        # We evaluate the generated content, which is everything except the last turn
        previous_turns = self._format_turn_for_evaluation(turns[-2]) if len(turns) >= 2 else ""
        
        # Get the final turn (ground truth)
        final_turn = self._format_turn_for_evaluation(turns[-1])
        
        # Evaluate each dimension
        for dimension_name, dimension_info in self.JUDGE_DIMENSIONS.items():
            print(f"  Evaluating dimension: {dimension_name}")
            
            # Create prompt for this dimension
            prompt = self._create_judge_prompt(
                previous_turns=previous_turns,
                final_turn=final_turn,
                dimension_name=dimension_name,
                dimension_definition=dimension_info['definition'],
                questions=dimension_info['questions']
            )

            print(f" Prompt: {prompt}")
            
            # Get LLM response
            llm_response = self._get_judge_response(prompt)
            print(f" LLM Response: {llm_response}")
            
            # Parse the response
            parsed_scores = self._parse_judge_response(llm_response, dimension_info['questions'])
            print(f" Parsed Scores: {parsed_scores}")
            
            results[dimension_name] = {
                'raw_response': llm_response,
                'scores': parsed_scores,
                'average_score': np.mean(list(parsed_scores.values())) if parsed_scores else 0.0
            }
        
        # Calculate overall average
        all_scores = []
        for dim_result in results.values():
            if dim_result['scores']:
                all_scores.extend(dim_result['scores'].values())
        
        results['overall_average'] = np.mean(all_scores) if all_scores else 0.0
        
        return results
    
    def _format_conversation_history(self, turns: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        lines = []
        for i, turn in enumerate(turns):
            if 'u1' in turn:
                lines.append(f"u1: {turn['u1']}")
            if 'u2' in turn:
                lines.append(f"u2: {turn['u2']}")
        return "\n".join(lines) if lines else "No previous conversation."
    
    def _format_turn_for_evaluation(self, turn: Dict[str, str]) -> str:
        """Format a single turn for evaluation."""
        parts = []
        if 'u1' in turn:
            parts.append(f"u1: {turn['u1']}")
        if 'u2' in turn:
            parts.append(f"u2: {turn['u2']}")
        return "\n".join(parts)
    
    def _create_judge_prompt(
        self,
        previous_turns: str,
        final_turn: str,
        dimension_name: str,
        dimension_definition: str,
        questions: Dict[str, str]
    ) -> str:
        """
        Create the judge prompt based on the template.
        
        Args:
            previous_turns: The previous_turns to evaluate
            final_turn: The final turn of the dialogue
            dimension_name: Name of the dimension being evaluated
            dimension_definition: Definition of the dimension
            questions: Dict of questions for this dimension
            
        Returns:
            Formatted prompt string
        """
        # Format questions
        formatted_questions = []
        example_answer = []
        for i, (q_name, q_text) in enumerate(questions.items(), 1):
            formatted_questions.append(f"Q{i} ({q_name}): {q_text}")
            example_answer.append(f"Q{i}:yes/no")  # Example answer format
        questions_text = "\n".join(formatted_questions)
        example_answers_text = "\n".join(example_answer)
        
        # Use proper indentation without leading spaces on each line
        prompt = f"""
<TASK_OVERVIEW>
You will be given a short conversation between two individuals u1 and u2 between the tags <SHORT_CONVERSATION>. 
You will then be given a generated dialogue starting from the short conversation, between the tags <GENERATED_DIALOGUE>.
You will be given a set of instructions between the tags <INSTRUCTIONS>.
Your task is to evaluate the generated dialogue (provided between the tags <GENERATED_DIALOGUE>) according to a specific dimension (provided between the tags <DIMENSION_DEFINITION>), taking into account the starting short conversation (provided between the tags <SHORT_CONVERSATION>).
To evaluate the dimension provided between the tags <DIMENSION_DEFINITION>, answer the questions provided in <QUESTIONS>, strictly with 'yes' or 'no', following the instructions given in the tag <INSTRUCTIONS> and the response format specified between the tags <ANSWER_FORMAT>.

<DIMENSION_DEFINITION>
{dimension_name} - {dimension_definition}
</DIMENSION_DEFINITION>

<INSTRUCTIONS>
1. Carefully read the short conversation (between the tags <SHORT_CONVERSATION>) and the generated dialogue (between the tags <GENERATED_DIALOGUE>).
2. Review the given questions associated with the dimension "{dimension_name}".
3. For each question between the tags <QUESTIONS>, answer strictly with 'yes' or 'no'.
4. Base your answers on clear reasoning, but **do not include the reasoning in the output**.
4. Evaluate the generated dialogue, taking into account the short conversation.
5. Follow the answer format exactly as specified between the tags <ANSWER_FORMAT>.
</INSTRUCTIONS>

<SHORT_CONVERSATION>
{final_turn}
</SHORT_CONVERSATION>

<GENERATED_DIALOGUE>
{previous_turns}
{final_turn}
</GENERATED_DIALOGUE>

<QUESTIONS>
{questions_text}
</QUESTIONS>

<ANSWER_FORMAT>
{example_answers_text}
</ANSWER_FORMAT>

Provide your answers to each question following the specified Answer Format.
</TASK_OVERVIEW>

The answer is:
<ANSWER_FORMAT>
Q1:
"""
        return prompt
    
    def _get_judge_response(self, prompt: str) -> str:
        """
        Get response from the judge model.
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            Model response string
        """
        try:
            # Pass prompt directly as string, similar to dialogue_data_augmentor
            response = self.pipe(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            if isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], dict) and 'generated_text' in response[0]:
                    generated_text = f"Q1:{response[0]['generated_text'].strip()}"
                    print(f" Generated text: {generated_text}")
                    return generated_text
            
            print(f"⚠ Unexpected response structure")
            return ""
            
        except Exception as e:
            print(f"⚠ Error getting judge response: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _parse_judge_response(self, response: str, questions: Dict[str, str]) -> Dict[str, float]:
        """
        Parse the judge response to extract yes/no answers and convert to scores.
        
        Args:
            response: Raw response from the judge model
            questions: Dict of questions to parse answers for
            
        Returns:
            Dict mapping question names to scores (1.0 for yes, 0.0 for no)
        """
        scores = {}
        
        # Regex pattern per catturare 'yes' o 'no' anche tra <>, spazi, newline
        answer_pattern = re.compile(r'Q(\d+)\s*:\s*[\s<]*?(yes|no)', re.IGNORECASE)
        
        # Trova tutte le risposte
        matches = answer_pattern.findall(response)
        # matches -> lista di tuple (numero_domanda, risposta)

        for i, (q_name, _) in enumerate(questions.items(), 1):
            score = 0.0
            # Cerca la corrispondenza per questa domanda
            for num, ans in matches:
                if int(num) == i:
                    score = 1.0 if ans.lower() == 'yes' else 0.0
                    break
            scores[q_name] = score

        return scores