#!/usr/bin/env python

import json
import os
import logging
import sys
import argparse
from datetime import datetime
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def setup_logging(log_level=logging.INFO):
    """Set up logging for test_visualizer.py with configurable level"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"test_visualizer_{timestamp}.log")
        ]
    )
    return logging.getLogger(__name__)

# Default logger
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    dirs = ["test_results/visualizations"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_latest_results_file():
    """Get the path to the most recent test results file"""
    results_files = glob.glob("test_results/sentiment_test_results_*.json")
    if not results_files:
        logger.error("No test results files found!")
        return None
    
    # Sort by modification time (most recent first)
    latest_file = max(results_files, key=os.path.getmtime)
    logger.info(f"Using latest results file: {latest_file}")
    return latest_file

def get_latest_diagnostic_file():
    """Get the path to the most recent diagnostic data file"""
    diag_files = glob.glob("test_results/diagnostic_data_*.json")
    if not diag_files:
        logger.info("No diagnostic data files found!")
        return None
    
    # Sort by modification time (most recent first)
    latest_file = max(diag_files, key=os.path.getmtime)
    logger.info(f"Using latest diagnostic file: {latest_file}")
    return latest_file

def get_specific_results_file(timestamp):
    """Get a results file with a specific timestamp"""
    filepath = f"test_results/sentiment_test_results_{timestamp}.json"
    if os.path.exists(filepath):
        logger.info(f"Using specified results file: {filepath}")
        return filepath
    else:
        logger.error(f"Specified results file not found: {filepath}")
        return None

def get_specific_diagnostic_file(timestamp):
    """Get a diagnostic file with a specific timestamp"""
    filepath = f"test_results/diagnostic_data_{timestamp}.json"
    if os.path.exists(filepath):
        logger.info(f"Using specified diagnostic file: {filepath}")
        return filepath
    else:
        logger.warning(f"Specified diagnostic file not found: {filepath}")
        return None

def load_results_data(file_path):
    """Load test results data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results data: {str(e)}")
        return None

def load_diagnostic_data(file_path):
    """Load diagnostic data from JSON file"""
    if not file_path:
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading diagnostic data: {str(e)}")
        return None

def plot_overall_accuracy(results_data, output_dir):
    """Create a bar chart showing overall accuracy for each model"""
    models = list(results_data.keys())
    accuracies = [results_data[model]["overall_pass_rate"] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Overall Sentiment Analysis Accuracy by Model')
    plt.ylim(0, 100)
    
    # Add percentage labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_accuracy.png")
    plt.close()
    
    return f"{output_dir}/overall_accuracy.png"

def plot_category_accuracy(results_data, output_dir):
    """Create a grouped bar chart showing accuracy by category"""
    # Get all unique categories
    categories = set()
    for model_data in results_data.values():
        categories.update(model_data["category_pass_rates"].keys())
    categories = sorted(list(categories))
    
    # Extract data for plotting
    models = list(results_data.keys())
    data = []
    
    for model in models:
        model_rates = []
        for category in categories:
            if category in results_data[model]["category_pass_rates"]:
                model_rates.append(results_data[model]["category_pass_rates"][category]["pass_rate"])
            else:
                model_rates.append(0)  # Category not present for this model
        data.append(model_rates)
    
    # Create plot
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    for i, model_data in enumerate(data):
        offset = width * (i - len(models)/2 + 0.5)
        rects = ax.bar(x + offset, model_data, width, label=models[i])
        
        # Add labels on bars
        for j, rect in enumerate(rects):
            height = rect.get_height()
            if height > 0:  # Only add text if there's a value
                ax.annotate(f'{height:.0f}%',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90 if height < 20 else 0)
    
    # Add labels, title and legend
    ax.set_xlabel('Category')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Sentiment Analysis Accuracy by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)  # Give space for annotations
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_accuracy.png")
    plt.close()
    
    return f"{output_dir}/category_accuracy.png"

def plot_confidence_vs_accuracy(results_data, output_dir):
    """Create a line chart showing accuracy by confidence band"""
    plt.figure(figsize=(10, 6))
    
    for model in results_data:
        bands = []
        accuracies = []
        
        # Extract data
        for band, data in results_data[model]["confidence_bands"].items():
            if data["count"] > 0:
                bands.append(band)
                accuracies.append((data["correct"] / data["count"]) * 100)
        
        # Sort by band
        sorted_data = sorted(zip(bands, accuracies), key=lambda x: int(x[0].split('-')[0]))
        bands, accuracies = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot line
        plt.plot(bands, accuracies, marker='o', linestyle='-', label=model)
    
    plt.xlabel('Confidence Band (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Relationship Between Model Confidence and Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_accuracy.png")
    plt.close()
    
    return f"{output_dir}/confidence_accuracy.png"

def plot_error_analysis(results_data, output_dir):
    """Create a confusion matrix for each model"""
    confusion_images = {}
    
    for model_name, model_data in results_data.items():
        # Count prediction patterns
        confusion = {
            ('Positive', 'Positive'): 0,
            ('Positive', 'Negative'): 0,
            ('Positive', 'Neutral'): 0,
            ('Negative', 'Positive'): 0,
            ('Negative', 'Negative'): 0,
            ('Negative', 'Neutral'): 0,
            ('Neutral', 'Positive'): 0,
            ('Neutral', 'Negative'): 0,
            ('Neutral', 'Neutral'): 0,
        }
        
        # Collect data
        for result in model_data["test_results"]:
            expected = result["expected"]
            predicted = result["predicted"]
            confusion[(expected, predicted)] += 1
        
        # Prepare confusion matrix
        matrix = np.zeros((3, 3))
        labels = ['Positive', 'Negative', 'Neutral']
        for i, expected in enumerate(labels):
            for j, predicted in enumerate(labels):
                matrix[i, j] = confusion[(expected, predicted)]
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.xlabel('Predicted')
        plt.ylabel('Expected')
        
        # Add numbers
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, format(int(matrix[i, j]), 'd'),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{model_name}.png")
        plt.close()
        
        confusion_images[model_name] = f"{output_dir}/confusion_matrix_{model_name}.png"
    
    return confusion_images

def plot_processing_time(results_data, output_dir):
    """Create a bar chart of average processing time by model"""
    models = list(results_data.keys())
    times = [results_data[model]["avg_processing_time_ms"] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, times, color=['#3498db', '#e74c3c'])
    plt.xlabel('Model')
    plt.ylabel('Avg. Processing Time (ms)')
    plt.title('Average Processing Time by Model')
    
    # Add time labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f} ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/processing_time.png")
    plt.close()
    
    return f"{output_dir}/processing_time.png"

def plot_diagnostic_timing(diagnostic_data, output_dir):
    """Create a visualization of average timing by pipeline stage"""
    if not diagnostic_data:
        return None
    
    # Collect timing data across all test cases
    stage_timings = defaultdict(list)
    for model_name, model_tests in diagnostic_data.items():
        for test in model_tests:
            for stage, timing in test.get('diagnostic_data', {}).get('stages_timing_ms', {}).items():
                stage_name = stage.split(':')[0] if ':' in stage else stage
                stage_timings[stage_name].append(timing)
    
    # Calculate average times
    avg_times = {stage: sum(times)/len(times) for stage, times in stage_timings.items() if times}
    
    # Sort by average time (descending)
    sorted_stages = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
    stages = [item[0] for item in sorted_stages]
    times = [item[1] for item in sorted_stages]
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.barh(stages, times, color='#2ecc71')
    plt.xlabel('Average Time (ms)')
    plt.ylabel('Pipeline Stage')
    plt.title('Average Processing Time by Pipeline Stage')
    
    # Add time labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f} ms', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pipeline_timing.png")
    plt.close()
    
    return f"{output_dir}/pipeline_timing.png"

def generate_html_report(results_data, diagnostic_data, images, output_path):
    """Generate an HTML report with all visualizations"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate summary stats
    models = list(results_data.keys())
    overall_accuracy = sum(results_data[model]["overall_pass_rate"] for model in models) / len(models)
    
    # Get problem categories
    problem_categories = []
    for model_name, model_data in results_data.items():
        for category, data in model_data["category_pass_rates"].items():
            if data["pass_rate"] < 70:  # Consider anything below 70% as problematic
                problem_categories.append((category, data["pass_rate"], data["count"]))
    
    # Deduplicate and sort
    problem_categories = sorted(set(problem_categories), key=lambda x: x[1])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .visualization {{ margin: 20px 0; }}
            .visualization img {{ max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .problems {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Sentiment Analysis Test Results</h1>
        <p>Report generated on {timestamp}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Overall Accuracy:</strong> {overall_accuracy:.1f}%</p>
            <p><strong>Models Tested:</strong> {', '.join(models)}</p>
        </div>
        
        <div class="problems">
            <h2>Areas for Improvement</h2>
    """
    
    if problem_categories:
        html += "<table>"
        html += "<tr><th>Category</th><th>Pass Rate</th><th>Tests Passing</th></tr>"
        for category, pass_rate, count in problem_categories:
            html += f"<tr><td>{category}</td><td>{pass_rate:.1f}%</td><td>{count}</td></tr>"
        html += "</table>"
    else:
        html += "<p>No problematic categories identified!</p>"
    
    html += """
        </div>
        
        <h2>Visualizations</h2>
    """
    
    # Add all visualization images
    for title, image_path in images.items():
        if image_path:
            # Convert to relative path for HTML
            rel_path = os.path.relpath(image_path, os.path.dirname(output_path))
            html += f"""
            <div class="visualization">
                <h3>{title}</h3>
                <img src="{rel_path}" alt="{title}">
            </div>
            """
    
    html += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report generated at {output_path}")
    return output_path

def generate_json_report(results_data, diagnostic_data, output_path):
    """Generate a JSON report with analysis information"""
    # Calculate summary stats
    models = list(results_data.keys())
    overall_accuracy = sum(results_data[model]["overall_pass_rate"] for model in models) / len(models)
    
    # Get problem categories
    problem_categories = []
    for model_name, model_data in results_data.items():
        for category, data in model_data["category_pass_rates"].items():
            if data["pass_rate"] < 70:  # Consider anything below 70% as problematic
                problem_categories.append({
                    "category": category,
                    "pass_rate": data["pass_rate"],
                    "tests_passing": data["count"]
                })
    
    # Create report structure
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_summary": {
            "average_accuracy": overall_accuracy,
            "models_tested": models
        },
        "model_accuracy": {
            model: results_data[model]["overall_pass_rate"] 
            for model in models
        },
        "problem_categories": sorted(problem_categories, key=lambda x: x["pass_rate"]),
        "category_accuracy": {
            model: results_data[model]["category_pass_rates"]
            for model in models
        },
        "confidence_bands": {
            model: results_data[model]["confidence_bands"]
            for model in models
        }
    }
    
    # Add diagnostic summary if available
    if diagnostic_data:
        # Analyze processing times by stage
        stage_timings = defaultdict(list)
        for model_name, model_tests in diagnostic_data.items():
            for test in model_tests:
                for stage, timing in test.get('diagnostic_data', {}).get('stages_timing_ms', {}).items():
                    stage_name = stage.split(':')[0] if ':' in stage else stage
                    stage_timings[stage_name].append(timing)
        
        # Calculate average times
        avg_times = {stage: sum(times)/len(times) for stage, times in stage_timings.items() if times}
        
        report["pipeline_timing"] = {
            "average_stage_times_ms": avg_times
        }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"JSON report generated at {output_path}")
    return output_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate visualizations and reports from test results')
    
    # Input options
    parser.add_argument('--timestamp', type=str, help='Specific timestamp of test results to visualize')
    
    # Output options
    parser.add_argument('--format', choices=['html', 'json', 'both'], default='html',
                        help='Output format for the report (default: html)')
    parser.add_argument('--output-dir', type=str, help='Custom output directory for visualizations')
    
    # Debug options
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(log_level=logging.DEBUG if args.debug else logging.INFO)
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Get results files
        if args.timestamp:
            results_file = get_specific_results_file(args.timestamp)
            diagnostic_file = get_specific_diagnostic_file(args.timestamp)
        else:
            # Use latest files
            results_file = get_latest_results_file()
            diagnostic_file = get_latest_diagnostic_file()
            
        if not results_file:
            return False
        
        # Load data
        results_data = load_results_data(results_file)
        if not results_data:
            return False
        
        diagnostic_data = load_diagnostic_data(diagnostic_file)
        
        # Output directory for visualizations
        timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = f"test_results/visualizations/{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        images = {}
        images['Overall Accuracy'] = plot_overall_accuracy(results_data, output_dir)
        images['Category Accuracy'] = plot_category_accuracy(results_data, output_dir)
        images['Confidence vs Accuracy'] = plot_confidence_vs_accuracy(results_data, output_dir)
        
        # Add confusion matrices
        confusion_images = plot_error_analysis(results_data, output_dir)
        for model, img_path in confusion_images.items():
            images[f'Confusion Matrix - {model}'] = img_path
        
        images['Processing Time'] = plot_processing_time(results_data, output_dir)
        
        if diagnostic_data:
            images['Pipeline Timing'] = plot_diagnostic_timing(diagnostic_data, output_dir)
        
        # Generate reports based on requested format
        if args.format in ['html', 'both']:
            report_path = f"test_results/performance_report_{timestamp}.html"
            generate_html_report(results_data, diagnostic_data, images, report_path)
            logger.info(f"HTML report generated at {report_path}")
        
        if args.format in ['json', 'both']:
            json_path = f"test_results/performance_data_{timestamp}.json"
            generate_json_report(results_data, diagnostic_data, json_path)
            logger.info(f"JSON report generated at {json_path}")
        
        logger.info(f"All visualizations saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 