"""
Agronomist-in-the-Loop LLM Reasoning Engine — ICPA Cotton Boll Pipeline

Converts boll morphology measurements into structured management
recommendations using frontier and open-weight language models.
Supports cloud APIs (Gemini, GPT, Claude) and local models (transformers).
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Try importing inference backends
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available. Local model inference disabled.")
    TRANSFORMERS_AVAILABLE = False

# API client imports
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Agronomic domain knowledge system prompt
SYSTEM_PROMPT = """You are an expert agronomist specializing in cotton (Gossypium hirsutum) production. You analyze quantitative boll morphology data from UAV-based 3D reconstruction to provide precision management recommendations.

Your expertise includes:
- Cotton growth stages (vegetative, squaring, flowering, boll development, open boll, harvest)
- Plant growth regulator (mepiquat chloride) timing and dosing
- Harvest aid (defoliant, boll opener, desiccant) scheduling
- Water stress indicators and irrigation management
- Yield estimation from boll count and size data

Rules:
1. Base ALL recommendations strictly on the provided measurements. Never fabricate data.
2. Express uncertainty explicitly when measurements are ambiguous.
3. Provide specific, actionable recommendations with quantitative thresholds.
4. Flag any measurements that appear anomalous or require field verification.
5. Output must be valid JSON matching the specified schema.

Output Schema:
{
  "growth_assessment": {
    "stage": "string (e.g., 'late boll development')",
    "trajectory": "string (e.g., 'progressing normally', 'stagnating')",
    "maturity_estimate_days_to_harvest": "integer or null"
  },
  "management_recommendations": [
    {
      "action": "string",
      "priority": "high|medium|low",
      "timing": "string",
      "rationale": "string"
    }
  ],
  "risk_flags": [
    {
      "flag": "string",
      "severity": "high|medium|low",
      "suggested_verification": "string"
    }
  ],
  "confidence": {
    "level": "high|medium|low",
    "justification": "string"
  }
}"""


def format_morphology_report(data: Dict) -> str:
    """Format morphological measurements into a structured prompt."""
    report = f"""Cotton Field Morphology Report
================================
Field ID: {data.get('field_id', 'unknown')}
Condition: {data.get('condition', 'unknown')}
Capture Date: {data.get('capture_date', 'unknown')}

Boll Population:
  Total bolls detected: {data.get('boll_count', 'N/A')}
  Mean diameter: {data.get('mean_diameter_mm', 'N/A')} mm
  Diameter CV: {data.get('diameter_cv', 'N/A')}
  Diameter range: {data.get('min_diameter_mm', 'N/A')} - {data.get('max_diameter_mm', 'N/A')} mm

Growth Indicators:
  Maturity index: {data.get('maturity_index', 'N/A')}
  Growth stagnation: {data.get('growth_stagnation_pct', 'N/A')}%
  Open boll percentage: {data.get('open_boll_pct', 'N/A')}%

Spatial Distribution:
  Lower canopy: {data.get('canopy_distribution', {}).get('lower', 'N/A')}
  Middle canopy: {data.get('canopy_distribution', {}).get('middle', 'N/A')}
  Upper canopy: {data.get('canopy_distribution', {}).get('upper', 'N/A')}

Visibility Metrics:
  Mean visibility score: {data.get('mean_visibility', 'N/A')}
  Visibility improvement post-defoliation: {data.get('visibility_improvement', 'N/A')}

Morphological Consistency:
  Volume CV: {data.get('volume_cv', 'N/A')}
  Mean compactness: {data.get('mean_compactness', 'N/A')}
  Mean aspect ratio: {data.get('mean_aspect_ratio', 'N/A')}
"""
    return report


class AgronomistLLM:
    """
    LLM for cotton management reasoning.
    Supports frontier APIs (Gemini 2.5 Pro, GPT-4.1, Claude) and
    local open-weight models (GLM-4-9B, Gemma 3, Phi-4, Qwen2.5-VL).
    """
    
    # Known API-based models (matched by prefix)
    API_PROVIDERS = {
        'gemini': 'google',
        'gpt': 'openai',
        'claude': 'anthropic',
    }
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        quantization: str = "4bit",
        device: str = "mps",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        image_paths: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device
        self.image_paths = image_paths or []
        self._model = None
        self._tokenizer = None
        self._api_client = None
        
        # Determine if this is an API model or local model
        self.provider = self._detect_provider(model_name)
        
        if self.provider:
            self._init_api_client()
        elif TRANSFORMERS_AVAILABLE:
            self._load_model(quantization)
    
    def _detect_provider(self, model_name: str) -> Optional[str]:
        """Detect if model is API-based by name prefix."""
        name_lower = model_name.lower()
        for prefix, provider in self.API_PROVIDERS.items():
            if name_lower.startswith(prefix):
                return provider
        return None
    
    def _init_api_client(self):
        """Initialize API client for frontier models."""
        if self.provider == 'google' and GENAI_AVAILABLE:
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self._api_client = genai.GenerativeModel(self.model_name)
                logger.info(f"Initialized Gemini API client: {self.model_name}")
            else:
                logger.warning("GOOGLE_API_KEY not set. Gemini API unavailable.")
        
        elif self.provider == 'openai' and OPENAI_AVAILABLE:
            self._api_client = OpenAI()  # Uses OPENAI_API_KEY env var
            logger.info(f"Initialized OpenAI API client: {self.model_name}")
        
        elif self.provider == 'anthropic' and ANTHROPIC_AVAILABLE:
            self._api_client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
            logger.info(f"Initialized Anthropic API client: {self.model_name}")
        
        else:
            logger.warning(f"API client for {self.provider} not available. Install the SDK.")
    
    def _load_model(self, quantization: str):
        """Load model with optional quantization."""
        logger.info(f"Loading {self.model_name} ({quantization})...")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            
            load_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16,
            }
            
            if quantization == '4bit':
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization.")
                    load_kwargs['device_map'] = 'auto'
            else:
                load_kwargs['device_map'] = 'auto'
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )
            
            logger.info(f"Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
    
    def generate_recommendation(
        self,
        morphology_data: Dict,
        num_runs: int = 1
    ) -> List[Dict]:
        """
        Generate management recommendations from morphology data.
        
        Args:
            morphology_data: dict with boll measurements
            num_runs: number of independent runs for consistency check
        
        Returns:
            List of response dicts (one per run)
        """
        user_prompt = format_morphology_report(morphology_data)
        user_prompt += "\n\nAnalyze this report and provide management recommendations in the specified JSON format."
        
        results = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            if self._api_client is not None:
                response_text = self._api_inference(user_prompt)
            elif self._model is not None and self._tokenizer is not None:
                response_text = self._inference(user_prompt)
            else:
                response_text = self._stub_response(morphology_data)
            
            elapsed = time.time() - start_time
            
            # Parse JSON from response
            parsed = self._extract_json(response_text)
            
            results.append({
                'run': run,
                'raw_response': response_text,
                'parsed': parsed,
                'latency_seconds': elapsed,
                'json_valid': parsed is not None,
                'model': self.model_name,
                'provider': self.provider or 'local',
            })
        
        return results
    
    def _api_inference(self, user_prompt: str) -> str:
        """Run inference via cloud API (Gemini, GPT, Claude)."""
        try:
            if self.provider == 'google':
                # Gemini API — supports multimodal
                parts = [SYSTEM_PROMPT + "\n\n" + user_prompt]
                
                # Add images if this is a VLM and images are provided
                if self.image_paths:
                    from PIL import Image as PILImage
                    for img_path in self.image_paths[:50]:  # limit frames
                        try:
                            img = PILImage.open(img_path)
                            parts.insert(-1, img)
                        except Exception:
                            pass
                
                response = self._api_client.generate_content(
                    parts,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                )
                return response.text
            
            elif self.provider == 'openai':
                response = self._api_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                response = self._api_client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"API inference failed ({self.provider}): {e}")
            return json.dumps({"error": str(e)})
    
    def _inference(self, user_prompt: str) -> str:
        """Run local model inference via transformers."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        inputs = self._tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True
        )
        
        if hasattr(self._model, 'device'):
            inputs = inputs.to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
            )
        
        response = self._tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _stub_response(self, data: Dict) -> str:
        """Generate a template response when model is not available."""
        boll_count = data.get('boll_count', 0)
        diameter = data.get('mean_diameter_mm', 0)
        maturity = data.get('maturity_index', 0)
        
        # Deterministic stub based on input values
        if maturity > 0.8:
            stage = "late boll development / early open"
            action = "Apply defoliant within 5-7 days"
            priority = "high"
        elif maturity > 0.5:
            stage = "mid boll development"
            action = "Monitor boll opening progression"
            priority = "medium"
        else:
            stage = "early boll development"
            action = "Consider PGR application if vegetative growth excessive"
            priority = "low"
        
        return json.dumps({
            "growth_assessment": {
                "stage": stage,
                "trajectory": "progressing normally" if maturity > 0.3 else "slow",
                "maturity_estimate_days_to_harvest": max(0, int((1 - maturity) * 60))
            },
            "management_recommendations": [
                {
                    "action": action,
                    "priority": priority,
                    "timing": "next 7 days",
                    "rationale": f"Based on maturity index {maturity:.2f} and {boll_count} detected bolls"
                }
            ],
            "risk_flags": [],
            "confidence": {
                "level": "medium",
                "justification": "STUB RESPONSE — model not loaded. Replace with actual LLM inference."
            }
        }, indent=2)
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from model response text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON block from markdown
        import re
        json_match = re.search(r'```json\s*(.*?)```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end+1])
            except json.JSONDecodeError:
                pass
        
        logger.warning("Could not extract valid JSON from response.")
        return None


def generate_test_cases() -> List[Dict]:
    """Generate synthetic test cases for LLM benchmarking."""
    test_cases = [
        {
            "name": "normal_mid_season",
            "field_id": "TX-2025-001",
            "condition": "pre_defoliation",
            "boll_count": 1247,
            "mean_diameter_mm": 28.3,
            "diameter_cv": 0.18,
            "min_diameter_mm": 15.2,
            "max_diameter_mm": 38.7,
            "maturity_index": 0.55,
            "growth_stagnation_pct": 5.0,
            "open_boll_pct": 15.0,
            "mean_visibility": 0.45,
            "visibility_improvement": 0.0,
            "volume_cv": 0.22,
            "mean_compactness": 0.78,
            "mean_aspect_ratio": 1.15,
            "canopy_distribution": {"lower": 0.3, "middle": 0.5, "upper": 0.2}
        },
        {
            "name": "ready_for_harvest_aid",
            "field_id": "TX-2025-001",
            "condition": "pre_defoliation",
            "boll_count": 1456,
            "mean_diameter_mm": 32.1,
            "diameter_cv": 0.12,
            "min_diameter_mm": 22.5,
            "max_diameter_mm": 39.8,
            "maturity_index": 0.82,
            "growth_stagnation_pct": 25.0,
            "open_boll_pct": 65.0,
            "mean_visibility": 0.55,
            "visibility_improvement": 0.0,
            "volume_cv": 0.15,
            "mean_compactness": 0.82,
            "mean_aspect_ratio": 1.08,
            "canopy_distribution": {"lower": 0.25, "middle": 0.45, "upper": 0.3}
        },
        {
            "name": "water_stress_detected",
            "field_id": "TX-2025-002",
            "condition": "pre_defoliation",
            "boll_count": 823,
            "mean_diameter_mm": 22.1,
            "diameter_cv": 0.35,
            "min_diameter_mm": 10.5,
            "max_diameter_mm": 33.2,
            "maturity_index": 0.38,
            "growth_stagnation_pct": 40.0,
            "open_boll_pct": 5.0,
            "mean_visibility": 0.3,
            "visibility_improvement": 0.0,
            "volume_cv": 0.45,
            "mean_compactness": 0.65,
            "mean_aspect_ratio": 1.4,
            "canopy_distribution": {"lower": 0.15, "middle": 0.35, "upper": 0.5}
        },
        {
            "name": "post_defoliation_assessment",
            "field_id": "TX-2025-001",
            "condition": "post_defoliation",
            "boll_count": 1523,
            "mean_diameter_mm": 31.5,
            "diameter_cv": 0.14,
            "min_diameter_mm": 20.1,
            "max_diameter_mm": 40.2,
            "maturity_index": 0.85,
            "growth_stagnation_pct": 28.0,
            "open_boll_pct": 72.0,
            "mean_visibility": 0.82,
            "visibility_improvement": 0.34,
            "volume_cv": 0.16,
            "mean_compactness": 0.84,
            "mean_aspect_ratio": 1.06,
            "canopy_distribution": {"lower": 0.3, "middle": 0.4, "upper": 0.3}
        },
        {
            "name": "uneven_maturity",
            "field_id": "TX-2025-003",
            "condition": "pre_defoliation",
            "boll_count": 1102,
            "mean_diameter_mm": 26.8,
            "diameter_cv": 0.28,
            "min_diameter_mm": 12.3,
            "max_diameter_mm": 38.9,
            "maturity_index": 0.52,
            "growth_stagnation_pct": 15.0,
            "open_boll_pct": 30.0,
            "mean_visibility": 0.42,
            "visibility_improvement": 0.0,
            "volume_cv": 0.35,
            "mean_compactness": 0.72,
            "mean_aspect_ratio": 1.25,
            "canopy_distribution": {"lower": 0.4, "middle": 0.35, "upper": 0.25}
        },
    ]
    
    return test_cases


def main():
    """Run LLM reasoning on test cases."""
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        config = yaml.safe_load(f)
    
    llm_config = config['llm']
    
    # Initialize LLM — defaults to frontier model (Gemini 2.5 Pro)
    model_name = llm_config.get('primary_model', 'gemini-2.5-pro')
    llm = AgronomistLLM(
        model_name=model_name,
        max_tokens=llm_config.get('max_tokens', 4096),
        temperature=llm_config.get('temperature', 0.1),
    )
    
    # Generate and run test cases
    test_cases = generate_test_cases()
    all_results = []
    
    for tc in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Case: {tc['name']}")
        logger.info(f"{'='*60}")
        
        results = llm.generate_recommendation(
            tc,
            num_runs=llm_config.get('benchmark', {}).get('num_runs', 3)
        )
        
        for r in results:
            logger.info(f"  Run {r['run']}: latency={r['latency_seconds']:.2f}s, valid_json={r['json_valid']}")
            if r['parsed']:
                recs = r['parsed'].get('management_recommendations', [])
                for rec in recs[:2]:
                    logger.info(f"    → [{rec.get('priority', '?')}] {rec.get('action', 'N/A')}")
        
        all_results.append({
            'test_case': tc['name'],
            'results': results
        })
    
    # Save results
    output_path = os.path.join(str(repo_root), 'outputs', 'metrics', 'llm_reasoning_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
