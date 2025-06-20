"""
Utility functions for ebook2audio.

Provides logging setup, file handling, progress tracking, and other common utilities.
"""

import os
import sys
import time
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import glob

from rich.console import Console
from rich.progress import (
    Progress, 
    TaskID, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel, Panel
from rich.text import Text
from rich.logging import RichHandler
from loguru import logger


class JobStatus(Enum):
    """Status of a processing job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a processing job."""
    id: str
    input_file: Path
    output_file: Path
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


class ProgressManager:
    """Manages rich progress bars for various operations."""
    
    def __init__(self, console: Console):
        self.console = console
        self.progress: Optional[Progress] = None
        self.tasks: Dict[str, TaskID] = {}
        
    def start(self, description: str = "Processing...") -> None:
        """Start the progress display."""
        if self.progress is not None:
            return
            
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=10,
        )
        self.progress.start()
        
    def stop(self) -> None:
        """Stop the progress display."""
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.tasks.clear()
    
    def add_task(self, task_id: str, description: str, total: int) -> TaskID:
        """Add a new progress task."""
        if self.progress is None:
            self.start()
        
        task = self.progress.add_task(description, total=total)
        self.tasks[task_id] = task
        return task
    
    def update_task(self, task_id: str, advance: int = 1, description: str = None) -> None:
        """Update progress for a task."""
        if self.progress is None or task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        kwargs = {"advance": advance}
        if description:
            kwargs["description"] = description
            
        self.progress.update(task, **kwargs)
    
    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if self.progress is None or task_id not in self.tasks:
            return
        
        task = self.tasks[task_id] 
        self.progress.update(task, completed=True)


class BatchProcessor:
    """Handles batch processing of multiple files."""
    
    def __init__(self, console: Console, max_workers: int = 4):
        self.console = console
        self.max_workers = max_workers
        self.jobs: List[JobInfo] = []
        self.progress_manager = ProgressManager(console)
        
    def add_job(self, input_file: Path, output_file: Path) -> str:
        """Add a job to the batch."""
        job_id = f"job_{len(self.jobs):04d}"
        job = JobInfo(
            id=job_id,
            input_file=input_file,
            output_file=output_file
        )
        self.jobs.append(job)
        return job_id
    
    def get_job_status(self) -> Dict[str, int]:
        """Get count of jobs by status."""
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = sum(1 for job in self.jobs if job.status == status)
        return status_counts
    
    def show_job_summary(self) -> None:
        """Display a summary table of all jobs."""
        table = Table(title="Batch Processing Summary")
        
        table.add_column("Job ID", style="cyan", no_wrap=True)
        table.add_column("Input File", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Progress", style="blue")
        table.add_column("Duration", style="magenta")
        
        for job in self.jobs:
            duration_str = ""
            if job.duration:
                duration_str = f"{job.duration:.1f}s"
            
            progress_str = f"{job.progress:.1f}%"
            
            # Color status based on value
            status_color = {
                JobStatus.PENDING: "white",
                JobStatus.RUNNING: "yellow", 
                JobStatus.COMPLETED: "green",
                JobStatus.FAILED: "red",
                JobStatus.CANCELLED: "orange"
            }.get(job.status, "white")
            
            table.add_row(
                job.id,
                str(job.input_file.name),
                f"[{status_color}]{job.status.value}[/{status_color}]",
                progress_str,
                duration_str
            )
        
        self.console.print(table)
    
    async def process_jobs(self, processor_func: Callable) -> None:
        """Process all jobs with specified processor function."""
        if not self.jobs:
            self.console.print("[yellow]No jobs to process[/yellow]")
            return
        
        self.progress_manager.start("Batch Processing")
        
        # Add progress task
        batch_task = self.progress_manager.add_task(
            "batch", 
            "Processing files...", 
            len(self.jobs)
        )
        
        # Create semaphore for limiting concurrent jobs
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_job(job: JobInfo) -> None:
            """Process a single job."""
            async with semaphore:
                job.status = JobStatus.RUNNING
                job.start_time = time.time()
                
                try:
                    await processor_func(job)
                    job.status = JobStatus.COMPLETED
                    job.progress = 100.0
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    logger.error(f"Job {job.id} failed: {e}")
                finally:
                    job.end_time = time.time()
                    self.progress_manager.update_task("batch")
        
        # Process all jobs concurrently
        tasks = [process_single_job(job) for job in self.jobs]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.progress_manager.stop()
        
        # Show final summary
        self.show_job_summary()
        
        # Print final stats
        status_counts = self.get_job_status()
        self.console.print(f"\n[green]✓ Completed: {status_counts['completed']}[/green]")
        if status_counts['failed'] > 0:
            self.console.print(f"[red]✗ Failed: {status_counts['failed']}[/red]")


class FileUtils:
    """File handling utilities."""
    
    @staticmethod
    def find_ebook_files(directory: Path, patterns: List[str] = None) -> List[Path]:
        """Find all ebook files in a directory."""
        if patterns is None:
            patterns = ["*.pdf", "*.epub", "*.mobi", "*.azw", "*.azw3"]
        
        files = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
            files.extend(directory.glob(f"**/{pattern}"))  # Recursive
        
        return sorted(set(files))  # Remove duplicates and sort
    
    @staticmethod
    def find_audio_files(directory: Path, patterns: List[str] = None) -> List[Path]:
        """Find all audio files in a directory."""
        if patterns is None:
            patterns = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
        
        files = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
            files.extend(directory.glob(f"**/{pattern}"))  # Recursive
        
        return sorted(set(files))
    
    @staticmethod
    def get_safe_filename(filename: str) -> str:
        """Get a safe filename by removing/replacing invalid characters."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        
        return filename
    
    @staticmethod
    def ensure_output_dir(output_path: Path) -> Path:
        """Ensure output directory exists and return it."""
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)


class CheckpointManager:
    """Manages checkpoint files for resume functionality."""
    
    def __init__(self, checkpoint_dir: Path = None):
        self.checkpoint_dir = checkpoint_dir or Path.cwd() / ".ebook2audio_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, job_id: str, checkpoint_data: Dict[str, Any]) -> None:
        """Save checkpoint data for a job."""
        checkpoint_file = self.checkpoint_dir / f"{job_id}.json"
        
        checkpoint_data["timestamp"] = time.time()
        checkpoint_data["job_id"] = job_id
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"Saved checkpoint for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for job {job_id}: {e}")
    
    def load_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data for a job."""
        checkpoint_file = self.checkpoint_dir / f"{job_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint for job {job_id}: {e}")
            return None
    
    def delete_checkpoint(self, job_id: str) -> None:
        """Delete checkpoint file for a job."""
        checkpoint_file = self.checkpoint_dir / f"{job_id}.json"
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.debug(f"Deleted checkpoint for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint for job {job_id}: {e}")
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint job IDs."""
        checkpoint_files = self.checkpoint_dir.glob("*.json")
        return [f.stem for f in checkpoint_files]
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> None:
        """Clean up old checkpoint files."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
                    logger.debug(f"Cleaned up old checkpoint: {checkpoint_file.name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint {checkpoint_file}: {e}")


def setup_logging(level: str = "INFO", console: Console = None) -> None:
    """Set up logging with rich formatting."""
    if console is None:
        console = Console()
    
    # Remove default loguru handler
    logger.remove()
    
    # Add rich handler
    logger.add(
        RichHandler(console=console, rich_tracebacks=True),
        level=level,
        format="{message}",
        backtrace=True,
        diagnose=True
    )
    
    # Also log to file
    log_file = Path.home() / ".config" / "ebook2audio" / "ebook2audio.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        backtrace=True,
        diagnose=True
    )


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"