#!/usr/bin/env python

import os
import sys
import logging
import time
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """Set up logging for run_tests.py with configurable level"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / f"run_tests_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    dirs = ["logs", "test_results", "test_results/visualizations"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def run_tests(args):
    """Run the test_improvements.py script with specified options"""
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    ensure_directories()
    
    start_time = time.time()
    logger.info("Starting test run...")
    
    # Build command with appropriate arguments
    cmd = [sys.executable, "test_improvements.py"]
    
    # Add category filtering if specified
    if args.categories:
        cmd.extend(["--categories", args.categories])
    
    # Add model selection if specified
    if args.models:
        cmd.extend(["--models", args.models])
    
    # Add other options
    if args.debug:
        cmd.append("--debug")
    
    if args.no_color:
        cmd.append("--no-color")
        
    if args.compare and args.compare != "none":
        cmd.extend(["--compare", args.compare])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
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
        
        # Get the latest results file for analysis
        results_file = get_latest_results_file()
        if results_file:
            analyze_results(results_file, logger)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with exit code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    
    # Generate visualizations if requested
    if args.visualize:
        try:
            logger.info("Generating visualizations...")
            viz_cmd = [sys.executable, "test_visualizer.py"]
            
            if args.report_format:
                viz_cmd.extend(["--format", args.report_format])
                
            viz_result = subprocess.run(
                viz_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True
            )
            
            logger.info(viz_result.stdout.strip())
            
            # Open report in browser if requested
            if args.open_report:
                latest_report = get_latest_report()
                if latest_report:
                    open_in_browser(latest_report)
                    logger.info(f"Opened report in browser: {latest_report}")
                else:
                    logger.warning("Could not find a report to open")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Visualization failed with exit code: {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    return True

def get_latest_results_file():
    """Find the most recent test results file"""
    results_dir = Path("test_results")
    if not results_dir.exists():
        return None
    
    results_files = list(results_dir.glob("sentiment_test_results_*.json"))
    if not results_files:
        return None
    
    # Sort by modification time (most recent first)
    latest = max(results_files, key=lambda p: p.stat().st_mtime)
    return latest

def get_latest_report():
    """Find the most recent HTML report file"""
    results_dir = Path("test_results")
    if not results_dir.exists():
        return None
    
    report_files = list(results_dir.glob("performance_report_*.html"))
    if not report_files:
        return None
    
    # Sort by modification time (most recent first)
    latest = max(report_files, key=lambda p: p.stat().st_mtime)
    return latest

def analyze_results(results_file, logger):
    """Analyze test results and print a summary"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        logger.info("\n===== QUICK RESULTS SUMMARY =====")
        
        # Print overall accuracy for each model
        for model, data in results.items():
            pass_rate = data.get("overall_pass_rate", 0)
            logger.info(f"{model}: {pass_rate:.1f}% overall accuracy")
        
        # Find problem areas (categories with < 70% pass rate)
        problem_areas = []
        for model, data in results.items():
            for category, cat_data in data.get("category_pass_rates", {}).items():
                if cat_data.get("pass_rate", 0) < 70:
                    problem_areas.append((category, cat_data.get("pass_rate", 0), cat_data.get("count", "0/0")))
        
        if problem_areas:
            logger.info("\nCategories needing improvement:")
            # Remove duplicates and sort by pass rate (ascending)
            unique_problems = sorted(set(problem_areas), key=lambda x: x[1])
            for category, pass_rate, count in unique_problems:
                logger.info(f"  - {category}: {pass_rate:.1f}% ({count})")
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")

def open_in_browser(file_path):
    """Open a file in the default web browser"""
    import webbrowser
    file_url = Path(file_path).absolute().as_uri()
    webbrowser.open(file_url)

def main():
    parser = argparse.ArgumentParser(description='Run sentiment analysis tests with options')
    
    # Test selection options
    parser.add_argument('--categories', type=str, help='Comma-separated list of test categories to run (e.g., "negation,sarcasm")')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to test (e.g., "naive_bayes,logistic_regression")')
    
    # Visualization options
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations after tests')
    parser.add_argument('--report-format', choices=['html', 'json', 'both'], default='html', help='Format for test reports')
    parser.add_argument('--open-report', '-o', action='store_true', help='Open the report in a browser when finished')
    
    # Comparison options
    parser.add_argument('--compare', type=str, default='none', help='Compare against previous results (specify timestamp or "latest")')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    
    args = parser.parse_args()
    
    success = run_tests(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 