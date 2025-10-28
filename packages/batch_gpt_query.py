"""
OpenAI Batch API support for InfoGap pipeline.

This module provides batch processing capabilities to replace synchronous API calls
with batch requests, reducing costs by 50% and improving throughput.

Key benefits:
- 50% cost reduction on all batch API calls
- Higher rate limits (separate from synchronous API)
- Automatic retry handling
- Better resource utilization

Usage:
1. Collect all requests first (fact extraction or intersection labeling)
2. Submit batch job
3. Wait for completion (polling or webhook)
4. Process results

Batch API documentation: https://platform.openai.com/docs/guides/batch
"""

import json
import time
from openai import OpenAI
import loguru
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = loguru.logger


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: Dict = None


@dataclass
class BatchJob:
    """Tracks a batch job submission and results."""
    batch_id: str
    input_file_id: str
    status: str
    created_at: float
    request_count: int
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    completed_at: Optional[float] = None
    failed_count: int = 0
    

def _is_openrouter_client(client: OpenAI) -> bool:
    """Return True when the client is configured for OpenRouter endpoints."""

    base_url = getattr(client, "base_url", None)
    if not base_url and hasattr(client, "_custom_client"):
        base_url = getattr(client._custom_client, "base_url", None)

    if not base_url:
        return False

    return "openrouter.ai" in str(base_url).lower()


class BatchGPTQuery:
    """
    Manages batch API calls to OpenAI for the InfoGap pipeline.
    
    Features:
    - Batches fact extraction requests (step_generate_facts)
    - Batches intersection labeling requests (step_compute_info_gap_reasoning)
    - Handles job submission, monitoring, and result retrieval
    - Provides caching integration with existing pipeline
    """
    
    def __init__(self, client: OpenAI, cache_dir: str = "scratch/batch_cache"):
        """
        Initialize batch query manager.
        
        Args:
            client: OpenAI client instance
            cache_dir: Directory to store batch job metadata and results
        """
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if _is_openrouter_client(client):
            raise RuntimeError(
                "OpenRouter does not currently support batch endpoints; "
                "set use_batch=False to fall back to synchronous processing."
            )
        
    def create_fact_extraction_batch(
        self,
        paragraphs: List[str],
        lang_code: str,
        model_name: str = "gpt-5-mini",
        person_name: str = "",
    ) -> List[BatchRequest]:
        """
        Create batch requests for fact extraction from paragraphs.
        
        Args:
            paragraphs: List of paragraph texts to process
            lang_code: Language code (e.g., 'en', 'fr')
            model_name: GPT model to use
            person_name: Person name for tracking (optional)
            
        Returns:
            List of BatchRequest objects ready for submission
        """
        from packages.constants import ASK_GPT_FACT_EXTRACTION_PROMPTS
        
        if lang_code not in ASK_GPT_FACT_EXTRACTION_PROMPTS:
            raise ValueError(f"Unsupported language code: {lang_code}")
        
        batch_requests = []
        for idx, paragraph in enumerate(paragraphs):
            custom_id = f"fact_extract_{person_name}_{lang_code}_{idx}"
            input_prompt = ASK_GPT_FACT_EXTRACTION_PROMPTS[lang_code].format(paragraph=paragraph)
            
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": input_prompt}],
                "temperature": 0,
                "max_tokens": len(paragraph) + 1000,
            }
            
            # Handle model-specific parameters
            model_lower = (model_name or "").lower()
            if model_lower in ("gpt-5", "gpt-5-mini"):
                body["max_completion_tokens"] = body.pop("max_tokens")
            
            batch_requests.append(BatchRequest(
                custom_id=custom_id,
                body=body
            ))
            
        return batch_requests
    
    def create_intersection_labeling_batch(
        self,
        fact_pairs: List[Tuple[str, List[str], List[List[str]]]],
        src_lang_code: str,
        tgt_lang_code: str,
        model_name: str = "gpt-5-mini",
        person_name: str = "",
        tgt_person_name: str = "",
    ) -> List[BatchRequest]:
        """
        Create batch requests for intersection labeling.
        
        Args:
            fact_pairs: List of (fact_id, src_context, tgt_contexts) tuples
            src_lang_code: Source language code
            tgt_lang_code: Target language code
            model_name: GPT model to use
            person_name: Person name in source language
            tgt_person_name: Person name in target language
            
        Returns:
            List of BatchRequest objects ready for submission
        """
        from packages.constants import ASK_GPT_FACT_INTERSECTION_PROMPTS, LANG_MAPPINGS
        from packages.gpt_query import format_fact_context
        
        if src_lang_code not in ASK_GPT_FACT_INTERSECTION_PROMPTS:
            raise ValueError(f"Invalid source language code: {src_lang_code}")
        
        src_language_map = LANG_MAPPINGS.get(src_lang_code, {})
        tgt_language = src_language_map.get(tgt_lang_code, tgt_lang_code)
        
        batch_requests = []
        for fact_id, src_context, tgt_contexts in fact_pairs:
            custom_id = f"intersection_{person_name}_{src_lang_code}_{tgt_lang_code}_{fact_id}"
            
            formatted_source = format_fact_context(src_context, src_lang_code)
            
            if len(tgt_contexts) == 1:
                formatted_target = format_fact_context(tgt_contexts[0], tgt_lang_code)
            else:
                logger.warning(f"Multiple target contexts for {fact_id}, using first")
                formatted_target = format_fact_context(tgt_contexts[0], tgt_lang_code)
            
            name_to_use = person_name if src_lang_code == 'en' else tgt_person_name
            
            input_prompt = ASK_GPT_FACT_INTERSECTION_PROMPTS[src_lang_code].format(
                person_name=name_to_use,
                tgt_person_name=tgt_person_name,
                src_fact_context=formatted_source,
                tgt_fact_context=formatted_target,
                tgt_language=tgt_language
            )
            
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": input_prompt}],
                "temperature": 0,
                "max_tokens": len(src_context) + len(tgt_contexts) + 2000,
            }
            
            # Handle model-specific parameters
            model_lower = (model_name or "").lower()
            if model_lower in ("gpt-5", "gpt-5-mini"):
                body["max_completion_tokens"] = body.pop("max_tokens")
            
            batch_requests.append(BatchRequest(
                custom_id=custom_id,
                body=body
            ))
            
        return batch_requests
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> BatchJob:
        """
        Submit a batch of requests to OpenAI.
        
        Args:
            requests: List of BatchRequest objects
            description: Human-readable description of the batch
            metadata: Optional metadata to attach to the batch
            
        Returns:
            BatchJob object with tracking information
        """
        # Write requests to JSONL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file_path = self.cache_dir / f"batch_input_{timestamp}.jsonl"
        
        with open(input_file_path, 'w') as f:
            for request in requests:
                batch_line = {
                    "custom_id": request.custom_id,
                    "method": request.method,
                    "url": request.url,
                    "body": request.body
                }
                f.write(json.dumps(batch_line) + '\n')
        
        logger.info(f"Created batch input file: {input_file_path} with {len(requests)} requests")
        
        # Upload file to OpenAI
        with open(input_file_path, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose='batch'
            )
        
        logger.info(f"Uploaded batch file: {file_response.id}")
        
        # Create batch job
        batch_response = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata or {}
        )
        
        batch_job = BatchJob(
            batch_id=batch_response.id,
            input_file_id=file_response.id,
            status=batch_response.status,
            created_at=time.time(),
            request_count=len(requests)
        )
        
        # Save batch job metadata
        job_file = self.cache_dir / f"batch_job_{batch_response.id}.json"
        with open(job_file, 'w') as f:
            json.dump({
                'batch_id': batch_job.batch_id,
                'input_file_id': batch_job.input_file_id,
                'input_file_path': str(input_file_path),
                'status': batch_job.status,
                'created_at': batch_job.created_at,
                'request_count': batch_job.request_count,
                'description': description,
                'metadata': metadata
            }, f, indent=2)
        
        logger.info(f"Submitted batch job: {batch_job.batch_id} with {len(requests)} requests")
        logger.info(f"Batch status: {batch_job.status}")
        
        return batch_job
    
    def check_batch_status(self, batch_job: BatchJob) -> BatchJob:
        """
        Check the status of a batch job.
        
        Args:
            batch_job: BatchJob to check
            
        Returns:
            Updated BatchJob with current status
        """
        response = self.client.batches.retrieve(batch_job.batch_id)
        
        batch_job.status = response.status
        batch_job.output_file_id = getattr(response, 'output_file_id', None)
        batch_job.error_file_id = getattr(response, 'error_file_id', None)
        
        if response.status == 'completed':
            batch_job.completed_at = time.time()
            batch_job.failed_count = getattr(response.request_counts, 'failed', 0)
        
        return batch_job
    
    def wait_for_batch(
        self,
        batch_job: BatchJob,
        poll_interval: int = 60,
        timeout: int = 86400,  # 24 hours
    ) -> BatchJob:
        """
        Wait for a batch job to complete, polling periodically.
        
        Args:
            batch_job: BatchJob to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            
        Returns:
            Completed BatchJob
            
        Raises:
            TimeoutError: If batch doesn't complete within timeout
        """
        start_time = time.time()
        
        while True:
            batch_job = self.check_batch_status(batch_job)
            elapsed = time.time() - start_time
            
            if batch_job.status == 'completed':
                logger.info(f"Batch {batch_job.batch_id} completed in {elapsed:.1f}s")
                return batch_job
            
            if batch_job.status == 'failed':
                logger.error(f"Batch {batch_job.batch_id} failed")
                raise RuntimeError(f"Batch job failed: {batch_job.batch_id}")
            
            if elapsed > timeout:
                raise TimeoutError(f"Batch {batch_job.batch_id} timed out after {timeout}s")
            
            logger.info(f"Batch {batch_job.batch_id} status: {batch_job.status}, elapsed: {elapsed:.1f}s")
            time.sleep(poll_interval)
    
    def retrieve_batch_results(self, batch_job: BatchJob) -> Dict[str, Dict]:
        """
        Retrieve and parse results from a completed batch job.
        
        Args:
            batch_job: Completed BatchJob
            
        Returns:
            Dictionary mapping custom_id to response data
        """
        if batch_job.status != 'completed':
            raise ValueError(f"Batch {batch_job.batch_id} is not completed (status: {batch_job.status})")
        
        if not batch_job.output_file_id:
            raise ValueError(f"Batch {batch_job.batch_id} has no output file")
        
        # Download output file
        output_content = self.client.files.content(batch_job.output_file_id)
        
        # Save locally
        output_file_path = self.cache_dir / f"batch_output_{batch_job.batch_id}.jsonl"
        with open(output_file_path, 'wb') as f:
            f.write(output_content.content)
        
        logger.info(f"Downloaded batch results to: {output_file_path}")
        
        # Parse results
        results = {}
        with open(output_file_path, 'r') as f:
            for line in f:
                response_obj = json.loads(line)
                custom_id = response_obj['custom_id']
                
                if response_obj.get('error'):
                    logger.error(f"Error in {custom_id}: {response_obj['error']}")
                    results[custom_id] = {'error': response_obj['error']}
                else:
                    response_body = response_obj['response']['body']
                    results[custom_id] = {
                        'content': response_body['choices'][0]['message']['content'],
                        'tokens': response_body['usage']['total_tokens']
                    }
        
        logger.info(f"Parsed {len(results)} results from batch {batch_job.batch_id}")
        
        return results
    
    def process_fact_extraction_results(
        self,
        results: Dict[str, Dict],
        paragraphs: List[str],
    ) -> Tuple[List[str], int]:
        """
        Process batch results for fact extraction.
        
        Args:
            results: Results dict from retrieve_batch_results
            paragraphs: Original paragraphs (for ordering)
            
        Returns:
            Tuple of (list of fact strings, total tokens used)
        """
        from packages.steps.info_diff_steps import extract_fact_decomp_list
        
        responses = []
        total_tokens = 0
        
        for idx in range(len(paragraphs)):
            custom_id = f"fact_extract_{idx}"
            
            # Find matching result (handle prefix matching)
            result = None
            for key, value in results.items():
                if key.endswith(f"_{idx}"):
                    result = value
                    break
            
            if result and 'content' in result:
                responses.append(result['content'])
                total_tokens += result['tokens']
            else:
                logger.warning(f"No result for paragraph {idx}")
                responses.append(None)
        
        return responses, total_tokens
    
    def process_intersection_results(
        self,
        results: Dict[str, Dict],
        fact_pairs: List[Tuple[str, any, any]],
        src_lang_code: str = 'en',
    ) -> Tuple[Dict[str, str], int]:
        """
        Process batch results for intersection labeling.
        
        Args:
            results: Results dict from retrieve_batch_results
            fact_pairs: Original fact pairs (for custom_id matching)
            src_lang_code: Source language for normalization
            
        Returns:
            Tuple of (dict mapping fact_id to label, total tokens used)
        """
        labels = {}
        total_tokens = 0
        
        for fact_id, _, _ in fact_pairs:
            # Find matching result
            result = None
            for key, value in results.items():
                if fact_id in key:
                    result = value
                    break
            
            if result and 'content' in result:
                response_content = result['content']
                
                # Normalize Yes/No responses
                if src_lang_code == "ru":
                    response_content = response_content.replace("да", "yes").replace("нет", "no")
                elif src_lang_code == "he":
                    response_content = response_content.replace("כן", "yes").replace("לא", "no")
                
                labels[fact_id] = response_content
                total_tokens += result['tokens']
            else:
                logger.warning(f"No result for fact pair {fact_id}")
                labels[fact_id] = "error"
        
        return labels, total_tokens


def batch_ask_gpt_for_facts(
    client: OpenAI,
    model_name: str,
    paragraphs: List[str],
    lang_code: str,
    person_name: str = "",
    wait_for_completion: bool = True,
    poll_interval: int = 30,
    timeout: int = 3600,
) -> Tuple[List[str], int, Optional[BatchJob]]:
    """
    Batch version of ask_gpt_for_facts.
    
    Args:
        client: OpenAI client
        model_name: GPT model to use
        paragraphs: List of paragraphs to process
        lang_code: Language code
        person_name: Person name for tracking
        wait_for_completion: If True, wait for batch to complete. If False, return job immediately.
        
    Returns:
        Tuple of (responses, total_tokens, batch_job or None)
    """
    batch_manager = BatchGPTQuery(client)
    
    requests = batch_manager.create_fact_extraction_batch(
        paragraphs=paragraphs,
        lang_code=lang_code,
        model_name=model_name,
        person_name=person_name,
    )
    
    batch_job = batch_manager.submit_batch(
        requests=requests,
        description=f"Fact extraction for {person_name} ({lang_code})",
        metadata={'person_name': person_name, 'lang_code': lang_code}
    )
    
    if not wait_for_completion:
        logger.info(f"Batch job submitted: {batch_job.batch_id}. Call wait_for_batch() to complete.")
        return [], 0, batch_job
    
    # Wait for completion
    batch_job = batch_manager.wait_for_batch(
        batch_job,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    
    # Retrieve and process results
    results = batch_manager.retrieve_batch_results(batch_job)
    responses, total_tokens = batch_manager.process_fact_extraction_results(results, paragraphs)
    
    return responses, total_tokens, batch_job
