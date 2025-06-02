import pandas as pd
import subprocess
import time

def test_local_model_with_and_without_training(
    model_name="qwen3:1.7b",
    training_examples_file="ai_comparison_test/comparison_test_5431_training_examples.txt",
    test_samples_file="ai_comparison_test/comparison_test_5431_test_samples_with_training.txt",
    reference_file="ai_comparison_test/comparison_test_5431_reference.csv",
    sample_limit=30  # Limit samples for faster testing
):
    """Test local model with and without training examples"""
    
    def query_ollama(model, prompt):
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    # Load reference data
    ref_df = pd.read_csv(reference_file)
    if sample_limit:
        ref_df = ref_df.head(sample_limit)
    
    # Load training examples
    with open(training_examples_file, 'r') as f:
        training_text = f.read()
    
    # Test 1: With training examples
    print(f"Testing {model_name} WITH training examples...")
    trained_results = []
    
    for _, row in ref_df.iterrows():
        # Create prompt with training examples + test sample
        full_prompt = f"{training_text}\n\nNow classify this text:\n{row['text']}\n\nClassification:"
        
        response = query_ollama(model_name, full_prompt)
        if response:
            prediction = "Suicidal" if "suicidal" in response.lower() or "suicide" in response.lower() else "Depressed"
            trained_results.append({
                'sample_id': row['sample_id'],
                'ai_prediction': prediction,
                'true_label': row['true_label']
            })
        
        time.sleep(1)  # Avoid overwhelming the system
    
    # Test 2: Without training examples
    print(f"Testing {model_name} WITHOUT training examples...")
    untrained_results = []
    
    for _, row in ref_df.iterrows():
        # Create prompt without training examples
        simple_prompt = f"Analyze this text and determine if it indicates depression:\n{row['text']}\n\nClassification:"
        
        response = query_ollama(model_name, simple_prompt)
        if response:
            prediction = "Suicidal" if "suicidal" in response.lower() or "suicide" in response.lower() else "Depressed"
            untrained_results.append({
                'sample_id': row['sample_id'],
                'ai_prediction': prediction,
                'true_label': row['true_label']
            })
        
        time.sleep(1)
    
    # Save results
    model_safe_name = model_name.replace(":", "_")
    
    trained_df = pd.DataFrame(trained_results)
    untrained_df = pd.DataFrame(untrained_results)
    
    trained_df.to_csv(f"{model_safe_name}_trained_results.csv", index=False)
    untrained_df.to_csv(f"{model_safe_name}_untrained_results.csv", index=False)
    
    print(f"Results saved for {model_name}")
    print(f"Trained samples: {len(trained_results)}")
    print(f"Untrained samples: {len(untrained_results)}")
    
    return {
        'trained_file': f"{model_safe_name}_trained_results.csv",
        'untrained_file': f"{model_safe_name}_untrained_results.csv"
    }

# Test both models
if __name__ == "__main__":
    # Test DeepSeek
    deepseek_results = test_local_model_with_and_without_training(
        model_name="deepseek-v2:16b",
        sample_limit=50  # Test with 25 samples for speed
    )
    
    # Test MedLLaMA
    medllama_results = test_local_model_with_and_without_training(
        model_name="medllama2:7b",
        sample_limit=50  # Test with 25 samples for speed
    )
    
    # Test qwen3
    qwen_results = test_local_model_with_and_without_training(
        model_name="qwen3:1.7b",
        sample_limit=50  # Test with 25 samples for speed
    )
    
    print("All local model testing completed!")