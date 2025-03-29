#!/usr/bin/env python

import os
import sys
import logging
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Set up basic logging for run_tests.py"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"run_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    dirs = ["logs", "test_results", "test_results/visualizations"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
def run_tests(args):
    """Run the test_improvements.py script"""
    logger = setup_logging()
    ensure_directories()
    
    start_time = time.time()
    logger.info("Starting test run...")
    
    # Run the tests
    try:
        cmd = [sys.executable, "test_improvements.py"]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        
        if args.verbose:
            logger.info("Test output:")
            logger.info(result.stdout)
            
            if result.stderr:
                logger.info("Test errors:")
                logger.info(result.stderr)
        
        logger.info("Tests completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    
    # Generate visualizations if requested
    if args.visualize:
        try:
            logger.info("Generating visualizations...")
            viz_cmd = [sys.executable, "test_visualizer.py"]
            viz_result = subprocess.run(
                viz_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True
            )
            
            logger.info(viz_result.stdout.strip())
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Visualization failed with exit code: {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run sentiment analysis tests')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    success = run_tests(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 