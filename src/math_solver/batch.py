"""Professional batch processing system for the Math Solver.

This module provides robust batch processing capabilities with comprehensive
logging, progress tracking, resume functionality, and error recovery.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import pandas as pd
from dataclasses import dataclass, asdict
import logging

from .api import solve_math_problem, get_usage_statistics
from .exceptions import MathSolverError, InputValidationError
from .logging_system import conversation_logger


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    
    batch_id: str
    total_items: int
    start_time: str
    checkpoint_interval: int = 10
    stats_interval: int = 25
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    backup_results: bool = True
    ignore_history: bool = True
    progress_callback: Optional[Callable] = None


@dataclass
class BatchResult:
    """Individual result from batch processing."""
    
    index: int
    problem: str
    solution: Optional[Dict[str, Any]] = None
    ground_truth: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    conversation_id: Optional[str] = None
    processing_time: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
    retry_count: int = 0
    timestamp: str = ""


class BatchProcessor:
    """Professional batch processing system with resume capability."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize the batch processor.
        
        Args:
            results_dir: Directory to store batch processing results.
                        Defaults to 'batch_results' in current directory.
        """
        self.results_dir = results_dir or Path.cwd() / "batch_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger = conversation_logger
        
        # Setup batch-specific logger
        self.batch_logger = logging.getLogger("math_solver.batch")
        self.batch_logger.setLevel(logging.INFO)
        
        # Add handler if not already present
        if not self.batch_logger.handlers:
            handler = logging.FileHandler(self.results_dir / "batch_processing.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.batch_logger.addHandler(handler)
    
    def _save_checkpoint(
        self, 
        config: BatchProcessingConfig, 
        results: List[BatchResult],
        current_index: int
    ) -> None:
        """Save processing checkpoint for resume capability.
        
        Args:
            config: Batch processing configuration.
            results: Current results list.
            current_index: Current processing index.
        """
        checkpoint_data = {
            'config': asdict(config),
            'results': [asdict(result) for result in results],
            'current_index': current_index,
            'checkpoint_time': datetime.now().isoformat(),
            'total_processed': len([r for r in results if r.success or r.error_message])
        }
        
        checkpoint_file = self.results_dir / f"checkpoint_{config.batch_id}.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.batch_logger.info(
                f"Checkpoint saved: {current_index}/{config.total_items} processed"
            )
        except Exception as e:
            self.batch_logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load processing checkpoint for resume.
        
        Args:
            batch_id: Batch ID to resume.
            
        Returns:
            Checkpoint data if found, None otherwise.
        """
        checkpoint_file = self.results_dir / f"checkpoint_{batch_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.batch_logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _save_final_results(
        self, 
        config: BatchProcessingConfig, 
        results: List[BatchResult]
    ) -> Path:
        """Save final batch processing results.
        
        Args:
            config: Batch processing configuration.
            results: Final results list.
            
        Returns:
            Path to the saved results file.
        """
        # Create comprehensive results data
        final_data = {
            'batch_metadata': {
                'batch_id': config.batch_id,
                'total_items': config.total_items,
                'start_time': config.start_time,
                'end_time': datetime.now().isoformat(),
                'success_count': len([r for r in results if r.success]),
                'failure_count': len([r for r in results if not r.success]),
                'total_processing_time': sum(r.processing_time for r in results),
                'total_tokens_used': sum(
                    r.token_usage.get('total_tokens', 0) 
                    for r in results if r.token_usage
                )
            },
            'results': [asdict(result) for result in results]
        }
        
        # Save as JSON
        results_file = self.results_dir / f"batch_results_{config.batch_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        # Also save as CSV for easy analysis
        csv_file = self.results_dir / f"batch_results_{config.batch_id}.csv"
        results_df = pd.DataFrame([asdict(result) for result in results])
        results_df.to_csv(csv_file, index=False)
        
        self.batch_logger.info(f"Final results saved to {results_file} and {csv_file}")
        
        return results_file
    
    def _log_progress_stats(self, config: BatchProcessingConfig, results: List[BatchResult]) -> None:
        """Log detailed progress statistics.
        
        Args:
            config: Batch processing configuration.
            results: Current results list.
        """
        processed_results = [r for r in results if r.success or r.error_message]
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if r.error_message]
        
        total_tokens = sum(
            r.token_usage.get('total_tokens', 0) 
            for r in successful_results if r.token_usage
        )
        
        avg_processing_time = (
            sum(r.processing_time for r in processed_results) / len(processed_results)
            if processed_results else 0
        )
        
        # Get overall usage statistics
        try:
            overall_stats = get_usage_statistics(days=1)  # Today's stats
        except Exception:
            overall_stats = {}
        
        self.batch_logger.info(
            f"BATCH PROGRESS [{config.batch_id}]: "
            f"{len(processed_results)}/{config.total_items} processed "
            f"({len(successful_results)} success, {len(failed_results)} failed) | "
            f"Tokens: {total_tokens:,} | "
            f"Avg time: {avg_processing_time:.2f}s | "
            f"Today total tokens: {overall_stats.get('total_tokens', 0):,}"
        )
        
        print(f"📊 Progress: {len(processed_results)}/{config.total_items} "
              f"({(len(processed_results)/config.total_items*100):.1f}%) | "
              f"✅ {len(successful_results)} | ❌ {len(failed_results)} | "
              f"🪙 {total_tokens:,} tokens")
    
    def process_batch(
        self,
        data: pd.DataFrame,
        problem_column: str = "problem",
        answer_column: str = "answer",
        image_column: Optional[str] = None,
        batch_id: Optional[str] = None,
        checkpoint_interval: int = 10,
        stats_interval: int = 25,
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
        ignore_history: bool = True,
        resume: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[BatchResult]:
        """Process a batch of math problems with professional error handling.
        
        Args:
            data: DataFrame with math problems.
            problem_column: Name of column containing problems.
            answer_column: Name of column containing ground truth answers.
            image_column: Optional column containing image paths.
            batch_id: Optional batch identifier for tracking.
            checkpoint_interval: Save checkpoint every N items.
            stats_interval: Log statistics every N items.
            max_retries: Maximum retry attempts per item.
            retry_delay_seconds: Delay between retries.
            ignore_history: Whether to ignore conversation history.
            resume: Whether to attempt resuming from checkpoint.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            List of BatchResult objects.
            
        Raises:
            MathSolverError: If batch processing fails critically.
        """
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Validate input data
        if data.empty:
            raise InputValidationError("Input DataFrame cannot be empty")
        
        if problem_column not in data.columns:
            raise InputValidationError(f"Problem column '{problem_column}' not found in DataFrame")
        
        if answer_column not in data.columns:
            raise InputValidationError(f"Answer column '{answer_column}' not found in DataFrame")
        
        # Initialize configuration
        config = BatchProcessingConfig(
            batch_id=batch_id,
            total_items=len(data),
            start_time=datetime.now().isoformat(),
            checkpoint_interval=checkpoint_interval,
            stats_interval=stats_interval,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            ignore_history=ignore_history,
            progress_callback=progress_callback
        )
        
        # Attempt to resume from checkpoint
        results = []
        start_index = 0
        
        if resume:
            checkpoint_data = self._load_checkpoint(batch_id)
            if checkpoint_data:
                self.batch_logger.info(f"Resuming batch {batch_id} from checkpoint")
                print(f"🔄 Resuming batch {batch_id} from checkpoint...")
                
                # Reconstruct results from checkpoint
                results = [
                    BatchResult(**result_data) 
                    for result_data in checkpoint_data['results']
                ]
                start_index = checkpoint_data['current_index']
                
                print(f"📍 Resuming from index {start_index} ({len(results)} already processed)")
        
        if start_index == 0:
            self.batch_logger.info(f"Starting new batch processing: {batch_id}")
            print(f"🚀 Starting batch processing: {batch_id}")
            print(f"📊 Total items to process: {config.total_items}")
        
        # Initialize tracking variables
        start_time = time.time()
        last_stats_index = 0
        
        try:
            # Process each row
            for i in range(start_index, len(data)):
                row = data.iloc[i]
                current_result = None
                
                # Find existing result if resuming
                if i < len(results):
                    current_result = results[i]
                    if current_result.success:
                        continue  # Skip already processed successful items
                
                # Initialize new result if needed
                if current_result is None:
                    current_result = BatchResult(
                        index=i,
                        problem=str(row[problem_column]),
                        ground_truth=str(row[answer_column]) if pd.notna(row[answer_column]) else None,
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(current_result)
                
                # Process image paths if column exists
                image_paths = None
                if image_column and image_column in data.columns:
                    image_value = row[image_column]
                    if pd.notna(image_value):
                        # Handle single path or comma-separated paths
                        if isinstance(image_value, str):
                            image_paths = [p.strip() for p in image_value.split(',') if p.strip()]
                        else:
                            image_paths = [str(image_value)]
                
                # Attempt to solve with retries
                retry_count = 0
                max_retry_attempts = max_retries if not current_result.success else max_retries - current_result.retry_count
                
                while retry_count <= max_retry_attempts:
                    try:
                        processing_start = time.time()
                        
                        # Solve the problem
                        solution = solve_math_problem(
                            problem=current_result.problem,
                            image_paths=image_paths,
                            ignore_history=config.ignore_history
                        )
                        
                        # Update result with success
                        current_result.solution = solution
                        current_result.success = True
                        current_result.conversation_id = solution['conversation_id']
                        current_result.processing_time = time.time() - processing_start
                        current_result.token_usage = solution.get('token_usage', {})
                        current_result.retry_count = retry_count
                        current_result.error_message = None
                        
                        self.batch_logger.info(
                            f"SUCCESS [{batch_id}] Item {i+1}/{config.total_items}: "
                            f"tokens={current_result.token_usage.get('total_tokens', 0)}, "
                            f"time={current_result.processing_time:.2f}s, "
                            f"retries={retry_count}"
                        )
                        
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        retry_count += 1
                        current_result.retry_count = retry_count
                        
                        if retry_count > max_retry_attempts:
                            # Max retries reached, mark as failed
                            current_result.success = False
                            current_result.error_message = str(e)
                            current_result.processing_time = time.time() - processing_start
                            
                            self.batch_logger.error(
                                f"FAILED [{batch_id}] Item {i+1}/{config.total_items}: "
                                f"{type(e).__name__}: {e} (after {retry_count} retries)"
                            )
                            
                            print(f"❌ Item {i+1} failed after {retry_count} retries: {e}")
                            break
                        else:
                            # Retry with delay
                            self.batch_logger.warning(
                                f"RETRY [{batch_id}] Item {i+1}/{config.total_items}: "
                                f"Attempt {retry_count}/{max_retry_attempts} failed: {e}"
                            )
                            
                            print(f"🔄 Item {i+1} retry {retry_count}/{max_retry_attempts} after error: {e}")
                            time.sleep(config.retry_delay_seconds)
                
                # Call progress callback if provided
                if config.progress_callback:
                    try:
                        config.progress_callback(i + 1, config.total_items, current_result)
                    except Exception as e:
                        self.batch_logger.warning(f"Progress callback failed: {e}")
                
                # Save checkpoint at intervals
                if (i + 1) % config.checkpoint_interval == 0:
                    self._save_checkpoint(config, results, i + 1)
                    print(f"💾 Checkpoint saved at item {i+1}")
                
                # Log statistics at intervals
                if (i + 1) % config.stats_interval == 0 or (i + 1 - last_stats_index) >= config.stats_interval:
                    self._log_progress_stats(config, results)
                    
                    # Get and log usage statistics
                    try:
                        usage_stats = get_usage_statistics(days=1)
                        self.batch_logger.info(
                            f"USAGE_STATS [{batch_id}]: "
                            f"Today total conversations: {usage_stats.get('total_conversations', 0)}, "
                            f"total tokens: {usage_stats.get('total_tokens', 0):,}"
                        )
                    except Exception as e:
                        self.batch_logger.warning(f"Failed to get usage statistics: {e}")
                    
                    last_stats_index = i + 1
                
                # Brief pause to be respectful to the API
                time.sleep(0.5)
            
            # Final checkpoint and statistics
            self._save_checkpoint(config, results, len(data))
            self._log_progress_stats(config, results)
            
            # Save final results
            results_file = self._save_final_results(config, results)
            
            # Log final summary
            total_time = time.time() - start_time
            successful_count = len([r for r in results if r.success])
            failed_count = len([r for r in results if not r.success])
            total_tokens = sum(
                r.token_usage.get('total_tokens', 0) 
                for r in results if r.token_usage
            )
            
            summary = {
                'batch_id': config.batch_id,
                'total_processed': len(results),
                'successful': successful_count,
                'failed': failed_count,
                'success_rate': f"{(successful_count/len(results)*100):.1f}%" if results else "0%",
                'total_time': f"{total_time:.2f}s",
                'avg_time_per_item': f"{total_time/len(results):.2f}s" if results else "0s",
                'total_tokens': f"{total_tokens:,}",
                'results_file': str(results_file)
            }
            
            self.batch_logger.info(f"BATCH_COMPLETE [{batch_id}]: {summary}")
            
            print(f"\n🎉 Batch processing completed!")
            print(f"📊 Results: {successful_count}✅ / {failed_count}❌ / {len(results)} total")
            print(f"⏱️  Total time: {total_time:.2f}s ({total_time/len(results):.2f}s avg)")
            print(f"🪙 Total tokens: {total_tokens:,}")
            print(f"💾 Results saved to: {results_file}")
            
            return results
            
        except KeyboardInterrupt:
            self.batch_logger.info(f"INTERRUPTED [{batch_id}]: Processing interrupted by user")
            print(f"\n⏸️  Processing interrupted. Progress saved to checkpoint.")
            
            # Save current progress
            self._save_checkpoint(config, results, len(results))
            
            print(f"🔄 To resume: batch_processor.resume_batch('{batch_id}')")
            return results
            
        except Exception as e:
            self.batch_logger.error(f"CRITICAL_ERROR [{batch_id}]: {e}")
            
            # Save current progress before failing
            try:
                self._save_checkpoint(config, results, len(results))
            except:
                pass
            
            raise MathSolverError(f"Batch processing failed: {e}")
    
    def resume_batch(self, batch_id: str) -> Optional[List[BatchResult]]:
        """Resume a previously interrupted batch.
        
        Args:
            batch_id: ID of the batch to resume.
            
        Returns:
            List of BatchResult objects if successful, None if no checkpoint found.
        """
        checkpoint_data = self._load_checkpoint(batch_id)
        
        if not checkpoint_data:
            print(f"❌ No checkpoint found for batch ID: {batch_id}")
            return None
        
        print(f"🔄 Resuming batch {batch_id}...")
        print(f"📍 Previously processed: {checkpoint_data['total_processed']} items")
        
        # Reconstruct the DataFrame and continue processing
        # Note: This requires the original DataFrame to be available
        # In a production system, you might want to save the DataFrame too
        print("⚠️  To resume, you need to call process_batch() again with resume=True and the same batch_id")
        
        return [BatchResult(**result_data) for result_data in checkpoint_data['results']]
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a batch processing job.
        
        Args:
            batch_id: ID of the batch to check.
            
        Returns:
            Status information if found, None otherwise.
        """
        checkpoint_data = self._load_checkpoint(batch_id)
        
        if not checkpoint_data:
            return None
        
        return {
            'batch_id': batch_id,
            'total_items': checkpoint_data['config']['total_items'],
            'processed_items': checkpoint_data['total_processed'],
            'current_index': checkpoint_data['current_index'],
            'progress_percentage': (checkpoint_data['total_processed'] / checkpoint_data['config']['total_items']) * 100,
            'last_checkpoint': checkpoint_data['checkpoint_time'],
            'start_time': checkpoint_data['config']['start_time']
        }
    
    def list_batches(self) -> List[Dict[str, Any]]:
        """List all available batch checkpoints.
        
        Returns:
            List of batch information dictionaries.
        """
        batches = []
        
        for checkpoint_file in self.results_dir.glob("checkpoint_*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                batch_info = {
                    'batch_id': checkpoint_data['config']['batch_id'],
                    'total_items': checkpoint_data['config']['total_items'],
                    'processed_items': checkpoint_data['total_processed'],
                    'progress_percentage': (checkpoint_data['total_processed'] / checkpoint_data['config']['total_items']) * 100,
                    'last_checkpoint': checkpoint_data['checkpoint_time'],
                    'checkpoint_file': str(checkpoint_file)
                }
                
                batches.append(batch_info)
                
            except Exception as e:
                self.batch_logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        return sorted(batches, key=lambda x: x['last_checkpoint'], reverse=True)


# Global batch processor instance
batch_processor = BatchProcessor()


def process_math_problems_batch(
    data: pd.DataFrame,
    problem_column: str = "problem",
    answer_column: str = "answer", 
    image_column: Optional[str] = None,
    batch_id: Optional[str] = None,
    **kwargs
) -> List[BatchResult]:
    """Convenient function for batch processing math problems.
    
    This is the main entry point for professional batch processing.
    
    Args:
        data: DataFrame containing math problems.
        problem_column: Name of column with problem text.
        answer_column: Name of column with ground truth answers.
        image_column: Optional column with image paths.
        batch_id: Optional batch identifier.
        **kwargs: Additional arguments passed to BatchProcessor.process_batch()
        
    Returns:
        List of processing results.
        
    Example:
        ```python
        import pandas as pd
        from math_solver.batch import process_math_problems_batch
        
        # Load your data
        data = pd.read_csv("math_problems.csv")
        
        # Process with professional logging and resume capability
        results = process_math_problems_batch(
            data=data,
            problem_column="problem",
            answer_column="answer",
            checkpoint_interval=5,
            stats_interval=10,
            ignore_history=True
        )
        
        # Check results
        successful = [r for r in results if r.success]
        print(f"Success rate: {len(successful)}/{len(results)}")
        ```
    """
    return batch_processor.process_batch(
        data=data,
        problem_column=problem_column,
        answer_column=answer_column,
        image_column=image_column,
        batch_id=batch_id,
        **kwargs
    )
