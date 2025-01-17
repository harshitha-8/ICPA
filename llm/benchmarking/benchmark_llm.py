"""
LLM Benchmarking — ICPA Cotton Boll Pipeline

Comparative evaluation of frontier and open-weight LLMs for structured
agronomic reasoning from cotton boll morphology data.
Covers cloud APIs (Gemini 2.5 Pro, GPT-4.1, Claude) and local models.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from reasoning_engine.reasoning_engine import AgronomistLLM, generate_test_cases


# Expected JSON output schema for validation
EXPECTED_SCHEMA_KEYS = {
    'growth_assessment': {'stage', 'trajectory'},
    'management_recommendations': None,  # list
    'risk_flags': None,  # list
    'confidence': {'level', 'justification'},
}


def validate_json_schema(response: Dict) -> Dict:
    """Validate LLM response against expected schema."""
    errors = []
    warnings = []
    
    if not isinstance(response, dict):
        return {'valid': False, 'errors': ['Response is not a dict'], 'warnings': [], 'score': 0.0}
    
    # Check top-level keys
    for key, expected_subkeys in EXPECTED_SCHEMA_KEYS.items():
        if key not in response:
            errors.append(f"Missing key: {key}")
            continue
        
        if expected_subkeys and isinstance(response[key], dict):
            for subkey in expected_subkeys:
                if subkey not in response[key]:
                    warnings.append(f"Missing sub-key: {key}.{subkey}")
    
    # Check recommendations is a list
    recs = response.get('management_recommendations', [])
    if not isinstance(recs, list):
        errors.append("management_recommendations must be a list")
    elif len(recs) == 0:
        warnings.append("No management recommendations provided")
    else:
        for i, rec in enumerate(recs):
            if not isinstance(rec, dict):
                errors.append(f"Recommendation {i} is not a dict")
            else:
                for field in ['action', 'priority']:
                    if field not in rec:
                        warnings.append(f"Recommendation {i} missing '{field}'")
    
    # Score
    total_checks = 4 + len(EXPECTED_SCHEMA_KEYS)
    passed = total_checks - len(errors)
    score = passed / total_checks
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'score': score
    }


def compute_inter_run_agreement(runs: List[Dict]) -> Dict:
    """Compute agreement between multiple LLM runs on the same input."""
    parsed_runs = [r for r in runs if r.get('parsed') is not None]
    
    if len(parsed_runs) < 2:
        return {'agreement_rate': 0.0, 'num_valid_runs': len(parsed_runs)}
    
    # Compare growth stage assessments
    stages = []
    for r in parsed_runs:
        p = r['parsed']
        if 'growth_assessment' in p and 'stage' in p['growth_assessment']:
            stages.append(p['growth_assessment']['stage'].lower().strip())
    
    stage_agreement = 1.0
    if len(stages) >= 2:
        # Fraction of pairs that agree
        agreements = 0
        total = 0
        for i in range(len(stages)):
            for j in range(i+1, len(stages)):
                total += 1
                if stages[i] == stages[j]:
                    agreements += 1
        stage_agreement = agreements / total if total > 0 else 0.0
    
    # Compare recommendation priorities
    priority_sets = []
    for r in parsed_runs:
        p = r['parsed']
        recs = p.get('management_recommendations', [])
        priorities = tuple(sorted(
            (rec.get('priority', ''), rec.get('action', '')[:30])
            for rec in recs
        ))
        priority_sets.append(priorities)
    
    priority_agreement = 1.0
    if len(priority_sets) >= 2:
        agreements = sum(
            1 for i in range(len(priority_sets))
            for j in range(i+1, len(priority_sets))
            if priority_sets[i] == priority_sets[j]
        )
        total = len(priority_sets) * (len(priority_sets) - 1) // 2
        priority_agreement = agreements / total if total > 0 else 0.0
    
    # Compare confidence levels
    confidence_levels = []
    for r in parsed_runs:
        p = r['parsed']
        if 'confidence' in p and 'level' in p['confidence']:
            confidence_levels.append(p['confidence']['level'].lower())
    
    confidence_agreement = 1.0
    if len(confidence_levels) >= 2:
        agreements = sum(
            1 for i in range(len(confidence_levels))
            for j in range(i+1, len(confidence_levels))
            if confidence_levels[i] == confidence_levels[j]
        )
        total = len(confidence_levels) * (len(confidence_levels) - 1) // 2
        confidence_agreement = agreements / total if total > 0 else 0.0
    
    overall = np.mean([stage_agreement, priority_agreement, confidence_agreement])
    
    return {
        'overall_agreement': float(overall),
        'stage_agreement': float(stage_agreement),
        'priority_agreement': float(priority_agreement),
        'confidence_agreement': float(confidence_agreement),
        'num_valid_runs': len(parsed_runs),
    }


def detect_hallucinations(response: Dict, input_data: Dict) -> Dict:
    """
    Check for hallucinated claims not supported by input data.
    Simple heuristic: flag if response mentions specific numbers
    not present in the input.
    """
    flags = []
    
    if not isinstance(response, dict):
        return {'hallucination_flags': [], 'hallucination_score': 0.0}
    
    # Serialize response and input for comparison
    response_str = json.dumps(response).lower()
    
    # Check for fabricated field IDs
    field_id = input_data.get('field_id', '')
    if 'field' in response_str and field_id and field_id.lower() not in response_str:
        # Response mentions a field but not our field ID — possible hallucination
        flags.append("Response may reference incorrect field ID")
    
    # Check for invented boll counts
    boll_count = input_data.get('boll_count', -1)
    import re
    numbers_in_response = re.findall(r'\b(\d{3,})\b', response_str)
    for num_str in numbers_in_response:
        num = int(num_str)
        # If a large number appears that's not close to any input value
        input_values = [v for v in input_data.values() if isinstance(v, (int, float)) and v > 100]
        if not any(abs(num - v) < v * 0.1 for v in input_values):
            flags.append(f"Number {num} not grounded in input data")
    
    score = 1.0 - min(len(flags) / 5, 1.0)  # 0 flags = 1.0, ≥5 flags = 0.0
    
    return {
        'hallucination_flags': flags,
        'hallucination_score': float(score),
        'num_flags': len(flags)
    }


def benchmark_model(
    model_name: str,
    test_cases: List[Dict],
    num_runs: int = 3,
    **model_kwargs
) -> Dict:
    """Run full benchmark for a single model."""
    logger.info(f"\n{'#'*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'#'*60}")
    
    llm = AgronomistLLM(model_name=model_name, **model_kwargs)
    
    all_results = []
    
    for tc in test_cases:
        logger.info(f"\n  Test case: {tc['name']}")
        
        runs = llm.generate_recommendation(tc, num_runs=num_runs)
        
        # Schema validation
        schema_scores = []
        for r in runs:
            if r['parsed']:
                validation = validate_json_schema(r['parsed'])
                r['schema_validation'] = validation
                schema_scores.append(validation['score'])
        
        # Inter-run agreement
        agreement = compute_inter_run_agreement(runs)
        
        # Hallucination detection
        hallucination_scores = []
        for r in runs:
            if r['parsed']:
                hall = detect_hallucinations(r['parsed'], tc)
                r['hallucination'] = hall
                hallucination_scores.append(hall['hallucination_score'])
        
        case_result = {
            'test_case': tc['name'],
            'runs': runs,
            'agreement': agreement,
            'mean_latency': float(np.mean([r['latency_seconds'] for r in runs])),
            'mean_schema_score': float(np.mean(schema_scores)) if schema_scores else 0.0,
            'json_compliance_rate': float(np.mean([r['json_valid'] for r in runs])),
            'mean_hallucination_score': float(np.mean(hallucination_scores)) if hallucination_scores else 0.0,
        }
        
        all_results.append(case_result)
        
        logger.info(f"    Latency: {case_result['mean_latency']:.2f}s")
        logger.info(f"    JSON compliance: {case_result['json_compliance_rate']:.1%}")
        logger.info(f"    Agreement: {agreement['overall_agreement']:.1%}")
    
    # Aggregate
    summary = {
        'model': model_name,
        'num_test_cases': len(test_cases),
        'num_runs_per_case': num_runs,
        'overall_latency': float(np.mean([r['mean_latency'] for r in all_results])),
        'overall_json_compliance': float(np.mean([r['json_compliance_rate'] for r in all_results])),
        'overall_agreement': float(np.mean([r['agreement']['overall_agreement'] for r in all_results])),
        'overall_schema_score': float(np.mean([r['mean_schema_score'] for r in all_results])),
        'overall_hallucination_score': float(np.mean([r['mean_hallucination_score'] for r in all_results])),
        'results': all_results,
    }
    
    return summary


def main():
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        config = yaml.safe_load(f)
    
    llm_config = config['llm']
    benchmark_config = llm_config.get('benchmark', {})
    
    test_cases = generate_test_cases()
    num_runs = benchmark_config.get('num_runs', 3)
    
    # Candidate models to benchmark — Frontier (cloud) + Open-weight (local)
    candidates = [
        # --- Frontier (Cloud APIs) ---
        {
            'model_name': 'gemini-2.5-pro',
            'category': 'Frontier',
            'modality': 'F, T',
            'cost_per_1k': 0.00125,
        },
        {
            'model_name': 'gpt-4.1',
            'category': 'Frontier',
            'modality': 'T',
            'cost_per_1k': 0.002,
        },
        {
            'model_name': 'claude-opus-4',
            'category': 'Frontier',
            'modality': 'T',
            'cost_per_1k': 0.015,
        },
        # --- Open-weight (Local) ---
        {
            'model_name': 'THUDM/glm-4-9b-chat',
            'category': 'Open-weight',
            'modality': 'T',
            'cost_per_1k': 0.0,
            'quantization': '4bit',
            'device': 'mps',
        },
        {
            'model_name': 'google/gemma-3-12b-it',
            'category': 'Open-weight',
            'modality': 'T',
            'cost_per_1k': 0.0,
            'quantization': '4bit',
            'device': 'mps',
        },
        {
            'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'category': 'Open-weight',
            'modality': 'F, T',
            'cost_per_1k': 0.0,
            'quantization': '4bit',
            'device': 'mps',
        },
        {
            'model_name': 'microsoft/phi-4',
            'category': 'Open-weight',
            'modality': 'T',
            'cost_per_1k': 0.0,
            'quantization': '4bit',
            'device': 'mps',
        },
        {
            'model_name': 'lmms-lab/LLaVA-Video-7B-Qwen2',
            'category': 'Open-weight',
            'modality': 'F, T',
            'cost_per_1k': 0.0,
            'quantization': '4bit',
            'device': 'mps',
        },
    ]
    
    all_benchmarks = []
    
    for candidate in candidates:
        # Extract benchmark-relevant kwargs (remove metadata fields)
        model_kwargs = {k: v for k, v in candidate.items()
                       if k not in ('category', 'modality', 'cost_per_1k')}
        
        summary = benchmark_model(
            test_cases=test_cases,
            num_runs=num_runs,
            **model_kwargs
        )
        summary['category'] = candidate.get('category', 'Unknown')
        summary['modality'] = candidate.get('modality', 'T')
        summary['cost_per_1k'] = candidate.get('cost_per_1k', 0.0)
        all_benchmarks.append(summary)
    
    # Comparison table (publication-style)
    logger.info(f"\n{'='*110}")
    logger.info("BENCHMARK COMPARISON — Table 5: LLM Reasoning Evaluation on 20 Cotton Morphology Test Cases")
    logger.info(f"{'='*110}")
    logger.info(
        f"{'Category':<12} {'Model':<28} {'Modality':<8} "
        f"{'Consist.↑':>10} {'Schema↑':>10} {'Hallu.Free↑':>12} "
        f"{'Latency↓':>10} {'Cost/1K':>10}"
    )
    logger.info('-' * 110)
    
    for b in all_benchmarks:
        logger.info(
            f"{b['category']:<12} "
            f"{b['model']:<28} "
            f"{b['modality']:<8} "
            f"{b['overall_agreement']:>9.1%} "
            f"{b['overall_schema_score']:>9.2f} "
            f"{b['overall_hallucination_score']:>11.2f} "
            f"{b['overall_latency']:>8.2f}s "
            f"${b['cost_per_1k']:>8.4f}"
        )
    
    # Save
    output_path = os.path.join(str(repo_root), 'outputs', 'metrics', 'llm_benchmark.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_benchmarks, f, indent=2, default=str)
    
    logger.info(f"\nBenchmark results saved to {output_path}")


if __name__ == '__main__':
    main()
