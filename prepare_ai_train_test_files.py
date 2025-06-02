import pandas as pd
import numpy as np
import random
import os

def prepare_training_examples(
    training_csv_path="training-set.csv",
    num_training_examples=30,  # Total examples to show AI (half depressed, half not)
    num_test_samples=50,       # Samples to test AI on (without labels)
    output_dir="ai_comparison_test",
    text_column="selftext",
    label_column="is_suicide",
    positive_label=1,
    session_id=None
):
    """
    Create training examples and test samples for comparing AI performance
    with and without prior training
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if session_id is None:
        session_id = f"comparison_test_{random.randint(1000, 9999)}"
    
    print(f"Preparing training/test samples (Session: {session_id})...")
    
    # Load training data
    try:
        df = pd.read_csv(training_csv_path)
        print(f"Loaded dataset with {len(df)} entries")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Check columns
    if text_column not in df.columns or label_column not in df.columns:
        print(f"Error: Required columns not found")
        return None
    
    # Count available samples
    total_positive = len(df[df[label_column] == positive_label])
    total_negative = len(df[df[label_column] != positive_label])
    print(f"Available: {total_positive} depressed, {total_negative} non-depressed")
    
    # Select training examples (balanced)
    examples_per_class = num_training_examples // 2
    
    training_positive = df[df[label_column] == positive_label].sample(examples_per_class)
    training_negative = df[df[label_column] != positive_label].sample(examples_per_class)
    
    # Select test samples (balanced, from remaining data)
    remaining_positive = df[~df.index.isin(training_positive.index) & (df[label_column] == positive_label)]
    remaining_negative = df[~df.index.isin(training_negative.index) & (df[label_column] != positive_label)]
    
    test_samples_per_class = num_test_samples // 2
    test_positive = remaining_positive.sample(min(test_samples_per_class, len(remaining_positive)))
    test_negative = remaining_negative.sample(min(test_samples_per_class, len(remaining_negative)))
    
    # Combine and shuffle test samples
    test_samples = pd.concat([test_positive, test_negative]).sample(frac=1).reset_index(drop=True)
    
    print(f"Selected {len(training_positive) + len(training_negative)} training examples")
    print(f"Selected {len(test_samples)} test samples")
    
    # Create training examples file
    training_file = os.path.join(output_dir, f"{session_id}_training_examples.txt")
    with open(training_file, 'w') as f:
        f.write("# Training Examples for Depression Detection\n\n")
        f.write("These are examples of texts and their correct classifications:\n\n")
        
        # Combine and shuffle training examples too
        training_examples = pd.concat([training_positive, training_negative]).sample(frac=1).reset_index(drop=True)
        
        for i, (_, row) in enumerate(training_examples.iterrows()):
            label = "Suicidal" if row[label_column] == positive_label else "Depressed"
            f.write(f"Example {i+1}:\n")
            f.write(f"Text: {row[text_column]}\n")
            f.write(f"Classification: {label}\n\n")
    
    print(f"Training examples saved to: {training_file}")
    
    # Create test samples (without labels) for trained AI
    trained_test_file = os.path.join(output_dir, f"{session_id}_test_samples_with_training.txt")
    with open(trained_test_file, 'w') as f:
        f.write("# Test Samples (For AI That Has Seen Training Examples)\n\n")
        f.write("Based on the training examples you've seen, classify each text below.\n")
        f.write("Respond with ONLY 'Suicidal' or 'Depressed' for each sample.\n\n")
        
        for i, (_, row) in enumerate(test_samples.iterrows()):
            f.write(f"Sample T{i+1:03d}:\n")
            f.write(f"{row[text_column]}\n\n")
    
    print(f"Test samples (with training) saved to: {trained_test_file}")
    
    # Create test samples (without labels) for untrained AI
    untrained_test_file = os.path.join(output_dir, f"{session_id}_test_samples_no_training.txt")
    with open(untrained_test_file, 'w') as f:
        f.write("# Test Samples (For AI Without Training Examples)\n\n")
        f.write("Analyze each text below and determine if it shows suicidal ideation or depression.\n")
        f.write("Respond with ONLY 'Suicidal' or 'Depressed' for each sample.\n\n")
        
        for i, (_, row) in enumerate(test_samples.iterrows()):
            f.write(f"Sample T{i+1:03d}:\n")
            f.write(f"{row[text_column]}\n\n")
    
    print(f"Test samples (no training) saved to: {untrained_test_file}")
    
    # Create reference file with true labels
    reference_file = os.path.join(output_dir, f"{session_id}_reference.csv")
    test_data = pd.DataFrame({
        'sample_id': [f"T{i+1:03d}" for i in range(len(test_samples))],
        'text': test_samples[text_column],
        'true_label': test_samples[label_column].apply(
            lambda x: "Suicidal" if x == positive_label else "Depressed"
        )
    })
    test_data.to_csv(reference_file, index=False)
    print(f"Reference file saved to: {reference_file}")
    
    # Create results templates
    trained_results_template = os.path.join(output_dir, f"{session_id}_trained_results.csv")
    untrained_results_template = os.path.join(output_dir, f"{session_id}_untrained_results.csv")
    
    for template_file in [trained_results_template, untrained_results_template]:
        results_template = pd.DataFrame({
            'sample_id': test_data['sample_id'],
            'ai_prediction': [''] * len(test_data),
            'true_label': test_data['true_label']
        })
        results_template.to_csv(template_file, index=False)
    
    print(f"Results templates created")
    
    # Create instructions
    instructions_file = os.path.join(output_dir, f"{session_id}_instructions.txt")
    with open(instructions_file, 'w') as f:
        f.write(f"""
# AI Training vs No-Training Comparison Instructions

## Purpose
Compare AI performance with and without prior training examples.

## Files Created:
1. {session_id}_training_examples.txt - Training examples to show the AI
2. {session_id}_test_samples_with_training.txt - Test samples for trained AI
3. {session_id}_test_samples_no_training.txt - Test samples for untrained AI
4. {session_id}_trained_results.csv - Record predictions from trained AI
5. {session_id}_untrained_results.csv - Record predictions from untrained AI
6. {session_id}_reference.csv - True labels (don't show to AI)

## Process:

### For Trained AI Test:
1. Open a new chat with your AI (Claude, etc.)
2. Copy and paste the entire content of '{session_id}_training_examples.txt'
3. Let the AI read and acknowledge the examples
4. Then copy and paste '{session_id}_test_samples_with_training.txt'
5. Record the AI's predictions in '{session_id}_trained_results.csv'

### For Untrained AI Test:
1. Open a completely NEW chat with the SAME AI
2. Copy and paste ONLY '{session_id}_test_samples_no_training.txt'
3. Record the AI's predictions in '{session_id}_untrained_results.csv'

### Compare Results:
Use the metrics calculator to compare both result files.

This tests {len(test_data)} samples:
- {len(test_data[test_data['true_label'] == 'Depressed'])} actually depressed
- {len(test_data[test_data['true_label'] == 'Not depressed'])} actually not depressed

Training examples: {len(training_examples)} ({examples_per_class} of each class)
""")
    
    print(f"Instructions saved to: {instructions_file}")
    
    # Create metrics calculator
    metrics_script = os.path.join(output_dir, f"{session_id}_compare_results.py")
    with open(metrics_script, 'w') as f:
        f.write(f'''
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def calculate_and_compare_metrics(trained_csv, untrained_csv):
    """Compare performance of trained vs untrained AI"""
    
    # Load both result files
    trained_df = pd.read_csv(trained_csv)
    untrained_df = pd.read_csv(untrained_csv)
    
    results = {{}}
    
    for name, df in [("Trained", trained_df), ("Untrained", untrained_df)]:
        # Convert to binary
        y_true = [1 if label == "Depressed" else 0 for label in df['true_label']]
        y_pred = [1 if pred == "Depressed" else 0 for pred in df['ai_prediction']]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        results[name] = {{
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }}
        
        print(f"\\n=== {{name}} AI Results ===")
        print(f"Accuracy:  {{accuracy:.4f}}")
        print(f"Precision: {{precision:.4f}}")
        print(f"Recall:    {{recall:.4f}}")
        print(f"F1 Score:  {{f1:.4f}}")
        print(f"Confusion Matrix:\\n{{cm}}")
    
    # Compare improvement
    improvement = {{}}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        improvement[metric] = results['Trained'][metric] - results['Untrained'][metric]
        print(f"\\n{{metric.capitalize()}} improvement: {{improvement[metric]:+.4f}}")
    
    # Create comparison chart
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    trained_scores = [results['Trained'][m] for m in metrics]
    untrained_scores = [results['Untrained'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, untrained_scores, width, label='Untrained', alpha=0.8)
    plt.bar(x + width/2, trained_scores, width, label='Trained', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Trained vs Untrained AI Performance')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (trained, untrained) in enumerate(zip(trained_scores, untrained_scores)):
        plt.text(i - width/2, untrained + 0.01, f'{{untrained:.3f}}', ha='center', va='bottom')
        plt.text(i + width/2, trained + 0.01, f'{{trained:.3f}}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_comparison.png')
    plt.show()
    
    return results, improvement

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <trained_results.csv> <untrained_results.csv>")
        sys.exit(1)
    
    trained_file = sys.argv[1]
    untrained_file = sys.argv[2]
    
    results, improvement = calculate_and_compare_metrics(trained_file, untrained_file)
    
    print("\\n=== Summary ===")
    print("Training examples improved performance by:")
    for metric, value in improvement.items():
        print(f"  {{metric.capitalize()}}: {{value:+.4f}}")
''')
    
    print(f"Comparison script saved to: {metrics_script}")
    
    print(f"\n✅ All files created successfully!")
    print(f"📁 Check the '{output_dir}' directory")
    print(f"📄 Read '{session_id}_instructions.txt' for next steps")
    
    return {
        'session_id': session_id,
        'training_file': training_file,
        'trained_test_file': trained_test_file,
        'untrained_test_file': untrained_test_file,
        'reference_file': reference_file,
        'instructions_file': instructions_file,
        'comparison_script': metrics_script
    }

# Run the script
if __name__ == "__main__":
    prepare_training_examples(
        training_csv_path="training-set.csv",
        num_training_examples=30,  # 15 depressed + 15 non-depressed examples
        num_test_samples=50,       # 25 + 25 test samples
        text_column="selftext",
        label_column="is_suicide",
        positive_label=1
    )