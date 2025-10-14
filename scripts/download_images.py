#!/usr/bin/env python3
"""
Optimized image downloader with multiprocessing and progress tracking.
Downloads images from URLs in a parquet file and saves them as PNG.
"""

import argparse
import hashlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageDownloader:
    """Handles image downloading with optimizations."""
    
    def __init__(
        self, 
        output_dir: str,
        timeout: int = 30,
        max_retries: int = 3,
        chunk_size: int = 8192
    ):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded images
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            chunk_size: Chunk size for streaming downloads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        
        # Session with connection pooling for efficiency
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        # Keep connections alive for reuse
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=0  # We handle retries manually
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def generate_filename(self, url: str, index: int) -> str:
        """Generate a unique filename based on URL hash and index."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        return f"image_{index:08d}_{url_hash}.png"
    
    def download_single_image(
        self, 
        args: Tuple[int, str, str]
    ) -> Tuple[int, bool, Optional[str]]:
        """
        Download a single image with retry logic.
        
        Args:
            args: Tuple of (index, url, filename)
            
        Returns:
            Tuple of (index, success, error_message)
        """
        index, url, filename = args
        filepath = self.output_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            return index, True, None
        
        for attempt in range(self.max_retries):
            try:
                # Stream download for memory efficiency
                response = self.session.get(
                    url, 
                    timeout=self.timeout,
                    stream=True
                )
                response.raise_for_status()
                
                # Read image data
                image_data = BytesIO()
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        image_data.write(chunk)
                
                image_data.seek(0)
                
                # Open and convert image to PNG
                with Image.open(image_data) as img:
                    # Convert to RGB if necessary (e.g., RGBA, grayscale)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create white background for transparency
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as PNG with optimization
                    img.save(
                        filepath, 
                        'PNG',
                        optimize=True,
                        compress_level=6  # Balance between speed and compression
                    )
                
                return index, True, None
                
            except requests.exceptions.Timeout:
                error_msg = f"Timeout on attempt {attempt + 1}"
                if attempt == self.max_retries - 1:
                    return index, False, error_msg
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                if attempt == self.max_retries - 1:
                    return index, False, error_msg
                time.sleep(1 * (attempt + 1))
                
            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
                return index, False, error_msg
        
        return index, False, "Max retries exceeded"


def download_worker(args: Tuple[int, str, str, str, int, int, int]) -> Tuple[int, bool, Optional[str]]:
    """
    Worker function for multiprocessing.
    Creates a new downloader instance for each process.
    """
    index, url, filename, output_dir, timeout, max_retries, chunk_size = args
    
    downloader = ImageDownloader(
        output_dir=output_dir,
        timeout=timeout,
        max_retries=max_retries,
        chunk_size=chunk_size
    )
    
    return downloader.download_single_image((index, url, filename))


def download_images(
    parquet_file: str,
    output_dir: str,
    url_column: str = 'image_url',
    num_workers: int = 16,
    timeout: int = 30,
    max_retries: int = 3,
    chunk_size: int = 8192,
    limit: Optional[int] = None
):
    """
    Download images from URLs in a parquet file.
    
    Args:
        parquet_file: Path to parquet file containing URLs
        output_dir: Directory to save images
        url_column: Name of column containing URLs
        num_workers: Number of parallel workers
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        chunk_size: Chunk size for streaming downloads
        limit: Maximum number of images to download (None for all)
    """
    logger.info(f"Loading data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in parquet file")
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {limit} images")
    
    total = len(df)
    logger.info(f"Found {total} images to download")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filenames
    downloader = ImageDownloader(output_dir)
    filenames = [
        downloader.generate_filename(url, idx) 
        for idx, url in enumerate(df[url_column])
    ]
    
    # Check existing files
    existing = sum(1 for f in filenames if (output_path / f).exists())
    if existing > 0:
        logger.info(f"Found {existing} existing files, skipping...")
    
    # Prepare download tasks
    tasks = [
        (idx, url, filename, output_dir, timeout, max_retries, chunk_size)
        for idx, (url, filename) in enumerate(zip(df[url_column], filenames))
    ]
    
    # Download with multiprocessing and progress bar
    successful = 0
    failed = 0
    failed_indices = []
    
    logger.info(f"Starting download with {num_workers} workers")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(download_worker, task): task[0] 
            for task in tasks
        }
        
        # Process results with progress bar
        with tqdm(total=total, desc="Downloading images", unit="img") as pbar:
            for future in as_completed(futures):
                try:
                    idx, success, error_msg = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        failed_indices.append(idx)
                        if error_msg:
                            logger.debug(f"Failed to download image {idx}: {error_msg}")
                except Exception as e:
                    failed += 1
                    logger.error(f"Worker exception: {str(e)}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': successful,
                    'failed': failed,
                    'skipped': existing
                })
    
    # Summary
    logger.info("=" * 60)
    logger.info("Download Summary:")
    logger.info(f"  Total images:      {total}")
    logger.info(f"  Successfully downloaded: {successful}")
    logger.info(f"  Failed:           {failed}")
    logger.info(f"  Already existed:  {existing}")
    logger.info("=" * 60)
    
    # Save failed indices if any
    if failed_indices:
        failed_file = output_path / "failed_downloads.txt"
        with open(failed_file, 'w') as f:
            for idx in failed_indices:
                f.write(f"{idx}\n")
        logger.info(f"Failed indices saved to {failed_file}")
    
    # Create metadata file
    metadata = {
        'index': list(range(len(df))),
        'filename': filenames,
        'prompt': df['prompt'].tolist() if 'prompt' in df.columns else None,
        'url': df[url_column].tolist()
    }
    metadata_df = pd.DataFrame(metadata)
    metadata_file = output_path / "metadata.parquet"
    metadata_df.to_parquet(metadata_file, index=False)
    logger.info(f"Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Download images from URLs with multiprocessing and progress tracking'
    )
    parser.add_argument(
        '--parquet_file',
        type=str,
        help='Path to parquet file containing image URLs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='downloaded_images',
        help='Directory to save downloaded images (default: downloaded_images)'
    )
    parser.add_argument(
        '--url-column',
        type=str,
        default='image_url',
        help='Name of column containing URLs (default: image_url)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=120,
        help='Number of parallel workers (default: 240)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retry attempts (default: 3)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8192,
        help='Chunk size for streaming downloads (default: 8192)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of images to download (default: all)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Start download
    start_time = time.time()
    
    download_images(
        parquet_file=args.parquet_file,
        output_dir=args.output_dir,
        url_column=args.url_column,
        num_workers=args.workers,
        timeout=args.timeout,
        max_retries=args.max_retries,
        chunk_size=args.chunk_size,
        limit=args.limit
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
