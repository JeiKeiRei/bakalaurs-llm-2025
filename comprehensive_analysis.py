import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_all_results():
    """Analyze all model results and create comprehensive comparison"""
    
    # Define your models and their result files (based on your file list)
    models = {
        'DeepSeek-v2 (16B)': {
            'trained': 'deepseek-v2_16b_trained_results.csv',
            'untrained': 'deepseek-v2_16b_untrained_results.csv'
        },
        'MedLLaMA2 (7B)': {
            'trained': 'medllama2_7b_trained_results.csv',
            'untrained': 'medllama2_7b_untrained_results.csv'
        },
        'Qwen3 (1.7B)': {
            'trained': 'qwen3_1.7b_trained_results.csv',
            'untrained': 'qwen3_1.7b_untrained_results.csv'
        },
        'ChatGPT 4o mini': {
            'trained': 'chatgpt_4o_mini_trained_results.csv',
            'untrained': 'chatgpt_4o_mini_untrained_results.csv'
        }
    }
    
    # Load reference file with your session ID
    reference_file = 'ai_comparison_test/comparison_test_5431_reference.csv'
    
    all_results = []
    
    try:
        # Load reference data to get true labels
        reference_df = pd.read_csv(reference_file)
        print(f"Loaded reference file with {len(reference_df)} samples")
        
        # Process all model results
        for model_name, files in models.items():
            for condition in ['trained', 'untrained']:
                try:
                    df = pd.read_csv(files[condition])
                    print(f"Processing {model_name} ({condition}) - {len(df)} samples")
                    
                    # Ensure we have the same number of samples as reference
                    if len(df) != len(reference_df):
                        print(f"Warning: {files[condition]} has {len(df)} samples, reference has {len(reference_df)}")
                        # Take the minimum length to avoid errors
                        min_len = min(len(df), len(reference_df))
                        df = df.head(min_len)
                        ref_subset = reference_df.head(min_len)
                    else:
                        ref_subset = reference_df
                    
                    # Convert to binary format
                    y_true = [1 if label == "Depressed" else 0 for label in ref_subset['true_label']]
                    y_pred = [1 if pred == "Depressed" else 0 for pred in df['ai_prediction']]
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_true, y_pred, f"{model_name} ({condition})")
                    metrics['model'] = model_name
                    metrics['condition'] = condition
                    metrics['model_size'] = extract_model_size(model_name)
                    metrics['deployment'] = 'Local' if model_name != 'ChatGPT 4o mini' else 'Cloud'
                    metrics['sample_count'] = len(y_true)
                    
                    all_results.append(metrics)
                    
                except FileNotFoundError:
                    print(f"Warning: File {files[condition]} not found")
                except Exception as e:
                    print(f"Error processing {files[condition]}: {e}")
        
    except FileNotFoundError:
        print(f"Error: Reference file {reference_file} not found")
        return None
    
    if not all_results:
        print("No results to analyze!")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save comprehensive results
    results_df.to_csv('comprehensive_model_comparison.csv', index=False)
    print(f"\nComprehensive results saved to 'comprehensive_model_comparison.csv'")
    
    # Create visualizations
    create_comprehensive_visualizations(results_df)
    
    # Print summary
    print_summary_analysis(results_df)
    
    # Create detailed comparison table
    create_detailed_comparison_table(results_df)
    
    return results_df

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate performance metrics for a model"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Get confusion matrix values
    if cm.shape == (2, 2):
        tn, fp = cm[0]
        fn, tp = cm[1]
    else:
        # Handle edge cases where only one class is predicted
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 1:  # All predicted as positive
                tp = sum(y_true)
                fp = len(y_true) - sum(y_true)
            else:  # All predicted as negative
                tn = len(y_true) - sum(y_true)
                fn = sum(y_true)
    
    print(f"\n=== {model_name} ===")
    print(f"Precizitāte:  {accuracy:.4f}")
    print(f"Precīzums: {precision:.4f}")
    print(f"Jūtība:    {recall:.4f}")
    print(f"F1 Rezultāts:  {f1:.4f}")
    print(f"Pareizi 'Suicidal' rezultāti: {tp}, Nepareizi 'Suicidal' rezultāti: {fp}")
    print(f"Pareizi 'Depressed' rezultāti: {tn}, Nepareizi 'Depressed' rezultāti: {fn}")
    print(f"Apjukuma matrica:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def extract_model_size(model_name):
    """Extract model size from model name"""
    if '16B' in model_name:
        return '16B'
    elif '7B' in model_name:
        return '7B'
    elif '1.7B' in model_name:
        return '1.7B'
    elif 'ChatGPT' in model_name:
        return 'Unknown (Cloud)'
    else:
        return 'Unknown'

def create_comprehensive_visualizations(results_df):
    """Izveido visaptverošas vizualizācijas modeļu veiktspējas analīzei"""
    
    # Uzstāda grafiku stilu
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Veiktspējas salīdzinājums pēc apstākļiem (2x2 subplot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Modeļu veiktspējas salīdzinājums: Apmācīti vs Neapmācīti', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Precizitāte (Accuracy)', 'Precīzums (Precision)', 'Jutība (Recall)', 'F1 rezultāts']
    colors = ['#4CAF50', '#FF7043']  # Zaļš un oranžs krāsu kombinācija
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i//2, i%2]
        
        # Pārveido datus ērtākai attēlošanai
        pivot_df = results_df.pivot(index='model', columns='condition', values=metric)
        
        # Pārveido kolonnu nosaukumus latviešu valodā
        if 'trained' in pivot_df.columns and 'untrained' in pivot_df.columns:
            pivot_df = pivot_df.rename(columns={'trained': 'Apmācīti', 'untrained': 'Neapmācīti'})
        
        # Izveido grupētu stabiņu diagrammu
        pivot_df.plot(kind='bar', ax=ax, alpha=0.85, color=colors, width=0.75)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel('Rezultāts', fontsize=11)
        ax.set_xlabel('Modeļi', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(title='Apmācības stāvoklis', title_fontsize=10, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Pievieno vērtību etiķetes uz stabiņiem
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, rotation=0, padding=3)
    
    plt.tight_layout()
    plt.savefig('modeļu_veiktspējas_salīdzinājums.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Saglabāts: modeļu_veiktspējas_salīdzinājums.png")
    plt.show()
    
    # 2. Apmācības uzlabojumu analīze (karstuma karte)
    improvement_data = []
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        if len(model_data) == 2:  # Pastāv gan apmācīti, gan neapmācīti dati
            trained = model_data[model_data['condition'] == 'trained'].iloc[0]
            untrained = model_data[model_data['condition'] == 'untrained'].iloc[0]
            
            for metric, label in zip(metrics, metric_labels):
                improvement = trained[metric] - untrained[metric]
                improvement_data.append({
                    'model': model,
                    'metric': label,
                    'improvement': improvement,
                    'improvement_percent': (improvement / untrained[metric] * 100) if untrained[metric] > 0 else 0
                })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        improvement_pivot = improvement_df.pivot(index='model', columns='metric', values='improvement')
        
        plt.figure(figsize=(14, 10))
        
        # Izveido karstuma karti ar uzlabotu krāsu shēmu
        heatmap = sns.heatmap(improvement_pivot, annot=True, cmap='RdYlBu_r', center=0, 
                    fmt='.3f', cbar_kws={'label': 'Uzlabojums (Apmācīti - Neapmācīti)'},
                    linewidths=0.8, linecolor='white', 
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        plt.title('Apmācības ietekme uz modeļu veiktspēju', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Veiktspējas rādītāji', fontsize=14, fontweight='bold')
        plt.ylabel('Modeļi', fontsize=14, fontweight='bold')
        plt.xticks(rotation=15, fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        
        # Pievieno krāsu skalu aprakstu
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Uzlabojums (pozitīvs = labāk)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('apmācības_uzlabojumu_karte.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print("✅ Saglabāts: apmācības_uzlabojumu_karte.png")
        plt.show()
    
    # 3. Lokālo vs Mākoņa modeļu salīdzinājums
    if 'deployment' in results_df.columns:
        local_models = results_df[results_df['deployment'] == 'Local']
        cloud_models = results_df[results_df['deployment'] == 'Cloud']
        
        if not local_models.empty or not cloud_models.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Lokālie vs Mākoņa modeļi: Veiktspējas salīdzinājums', 
                        fontsize=16, fontweight='bold')
            
            deployment_colors = {
                'Lokālie (Apmācīti)': '#2E7D32',      # Tumši zaļš
                'Lokālie (Neapmācīti)': '#81C784',    # Gaiši zaļš  
                'Mākoņa (Apmācīti)': '#1565C0',       # Tumši zils
                'Mākoņa (Neapmācīti)': '#64B5F6'      # Gaiši zils
            }
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i//2, i%2]
                
                x_pos = 0
                bar_width = 0.35
                
                # Attēlo lokālos modeļus
                if not local_models.empty:
                    for j, condition in enumerate(['trained', 'untrained']):
                        subset = local_models[local_models['condition'] == condition]
                        if not subset.empty:
                            condition_lv = 'Apmācīti' if condition == 'trained' else 'Neapmācīti'
                            color_key = f'Lokālie ({condition_lv})'
                            
                            bars = ax.bar([x_pos + j * bar_width], subset[metric].iloc[0], 
                                         width=bar_width, alpha=0.8, 
                                         color=deployment_colors[color_key],
                                         label=color_key)
                            
                            # Pievieno vērtību uz stabiņa
                            ax.bar_label(bars, fmt='%.3f', fontsize=9, fontweight='bold')
                
                # Attēlo mākoņa modeļus
                if not cloud_models.empty:
                    x_pos = 1.0
                    for j, condition in enumerate(['trained', 'untrained']):
                        subset = cloud_models[cloud_models['condition'] == condition]
                        if not subset.empty:
                            condition_lv = 'Apmācīti' if condition == 'trained' else 'Neapmācīti'
                            color_key = f'Mākoņa ({condition_lv})'
                            
                            bars = ax.bar([x_pos + j * bar_width], subset[metric].iloc[0], 
                                         width=bar_width, alpha=0.8,
                                         color=deployment_colors[color_key],
                                         label=color_key)
                            
                            # Pievieno vērtību uz stabiņa
                            ax.bar_label(bars, fmt='%.3f', fontsize=9, fontweight='bold')
                
                ax.set_xlabel('Izvietošanas veids', fontsize=11)
                ax.set_ylabel(label.split(' ')[0], fontsize=11)
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.set_xticks([0.175, 1.175])
                ax.set_xticklabels(['Lokālie modeļi', 'Mākoņa modeļi'])
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_ylim(0, 1.05)
            
            plt.tight_layout()
            plt.savefig('lokālie_vs_mākoņa_salīdzinājums.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print("✅ Saglabāts: lokālie_vs_mākoņa_salīdzinājums.png")
            plt.show()
    
    # 4. Modeļa izmēra analīze (tikai lokālajiem modeļiem)
    # 4. Modeļa izmēra analīze (tikai lokālajiem modeļiem) - LABOTĀ VERSIJA
    if 'model_size' in results_df.columns and 'deployment' in results_df.columns:
        local_trained = results_df[(results_df['deployment'] == 'Local') & 
                                 (results_df['condition'] == 'trained')]
        
        if len(local_trained) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Modeļa izmēra ietekme uz veiktspēju (Lokālie apmācītie modeļi)', 
                        fontsize=16, fontweight='bold')
            
            # Kartē modeļu izmērus uz skaitliskām vērtībām
            # LABOTS: Qwen3 ir 1.7B, nevis 7B
            size_mapping = {
                '1.7B': 1.7, 
                '3B': 3.0, 
                '7B': 7.0, 
                '13B': 13.0, 
                '16B': 16.0, 
                '32B': 32.0
            }
            
            # Specifiskā kartēšana modeļiem ar nepareiziem nosaukumiem
            model_size_corrections = {
                'Qwen3 (1.7B)': '1.7B',  # Qwen3 faktiski ir 1.7B parametri
                'MedLLaMA2 (7B)': '7B',  # Ja nepieciešams
            }
            
            for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i//2, i%2]
                
                model_sizes = []
                model_scores = []
                model_labels = []  # LABOTS: mainīts nosaukums, lai nebūtu konflikts
                
                for _, row in local_trained.iterrows():
                    model_name = row['model']
                    stated_size = row['model_size']
                    
                    # Izmanto korekcijas, ja nepieciešams
                    if model_name in model_size_corrections:
                        actual_size = model_size_corrections[model_name]
                    else:
                        actual_size = stated_size
                    
                    if actual_size in size_mapping:
                        model_sizes.append(size_mapping[actual_size])
                        model_scores.append(row[metric])
                        
                        # Izveido skaidru etiķeti
                        clean_model_name = model_name.split(' ')[0] if ' ' in model_name else model_name
                        model_labels.append(f"{clean_model_name}\n({actual_size})")
                
                if model_sizes and len(model_sizes) >= 2:
                    # Izveido punktu diagrammu
                    scatter = ax.scatter(model_sizes, model_scores, alpha=0.8, s=200, 
                                       c=range(len(model_sizes)), cmap='plasma', 
                                       edgecolors='black', linewidth=1.5)
                    
                    # Pievieno modeļu etiķetes
                    for size, score, label_text in zip(model_sizes, model_scores, model_labels):
                        # Uzlabo etiķetes pozicionēšanu
                        offset_x = 15 if size < 10 else -15  # Mazākiem modeļiem pa labi, lielākiem pa kreisi
                        ha_align = 'left' if size < 10 else 'right'
                        
                        # Intelligent positioning to avoid title overlap
                        if score > 0.8:  # If point is high, place label below
                            offset_y = -25
                            va_align = 'top'
                        else:  # If point is lower, place label above
                            offset_y = 15
                            va_align = 'bottom'
                        
                        ax.annotate(label_text, (size, score), 
                                   xytext=(offset_x, offset_y), 
                                   textcoords='offset points', 
                                   fontsize=9,
                                   bbox=dict(boxstyle='round,pad=0.4', 
                                           facecolor='white', 
                                           alpha=0.9,
                                           edgecolor='gray'),
                                   ha=ha_align,
                                   va=va_align)
                    
                    # Pievieno tendences līniju
                    if len(model_sizes) >= 2:  # Samazināts no 3 uz 2, lai darbotos ar mazāk punktiem
                        try:
                            z = np.polyfit(model_sizes, model_scores, 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(min(model_sizes), max(model_sizes), 100)
                            
                            # Aprēķina R² vērtību
                            correlation_matrix = np.corrcoef(model_sizes, model_scores)
                            r_squared = correlation_matrix[0,1]**2
                            
                            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
                                   label=f'Tendence (R² = {r_squared:.3f})')
                            ax.legend(fontsize=9, loc='best')
                        except:
                            print(f"Nevarēja aprēķināt tendences līniju metrikai: {metric}")
                
                # Uzlabo ass etiķetes
                ax.set_xlabel('Modeļa izmērs (miljardi parametru)', fontsize=11, fontweight='bold')
                ax.set_ylabel(metric_label.split(' ')[0], fontsize=11, fontweight='bold')  # LABOTS: izmanto metric_label
                ax.set_title(metric_label, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Uzlabo ass iestatījumus
                if model_sizes:
                    ax.set_xscale('log')
                    ax.set_xlim(left=min(model_sizes)*0.7, right=max(model_sizes)*1.4)
                    
                    # Uzstāda x-ass etiķetes
                    x_ticks = sorted(list(set(model_sizes)))
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels([f'{x:.1f}B' for x in x_ticks])
                
                ax.set_ylim(0, 1.05)
                
                # Pievieno papildu informāciju, ja nav datu
                if not model_sizes:
                    ax.text(0.5, 0.5, 'Nav pieejami dati', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, alpha=0.6)
            
            plt.tight_layout()
            plt.savefig('modeļa_izmērs_vs_veiktspēja.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print("✅ Saglabāts: modeļa_izmērs_vs_veiktspēja.png")
            plt.show()
            
            # Papildu debug informācija
            print("\n🔍 DEBUG INFO - Modeļu izmēri:")
            for _, row in local_trained.iterrows():
                model_name = row['model']
                stated_size = row['model_size']
                corrected_size = model_size_corrections.get(model_name, stated_size)
                print(f"  • {model_name}: {stated_size} → {corrected_size}")
    
    # 5. Kopsavilkuma tabula
    print("\n" + "="*80)
    print("📊 MODEĻU VEIKTSPĒJAS KOPSAVILKUMS")
    print("="*80)
    
    # Aprēķina vidējos rādītājus
    summary_stats = results_df.groupby(['model', 'condition'])[metrics].mean()
    
    for model in results_df['model'].unique():
        print(f"\n🔹 {model.upper()}")
        print("-" * (len(model) + 4))
        
        model_data = results_df[results_df['model'] == model]
        
        for condition in ['trained', 'untrained']:
            condition_lv = "Apmācīts" if condition == 'trained' else "Neapmācīts"
            subset = model_data[model_data['condition'] == condition]
            
            if not subset.empty:
                print(f"  {condition_lv}:")
                for metric, label in zip(metrics, metric_labels):
                    value = subset[metric].iloc[0]
                    print(f"    • {label}: {value:.3f}")
        
        # Aprēķina uzlabojumu
        trained_data = model_data[model_data['condition'] == 'trained']
        untrained_data = model_data[model_data['condition'] == 'untrained']
        
        if not trained_data.empty and not untrained_data.empty:
            print("  📈 Uzlabojums:")
            for metric, label in zip(metrics, metric_labels):
                improvement = trained_data[metric].iloc[0] - untrained_data[metric].iloc[0]
                improvement_pct = (improvement / untrained_data[metric].iloc[0] * 100) if untrained_data[metric].iloc[0] > 0 else 0
                print(f"    • {label}: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    print("\n" + "="*80)
    print("✅ Visi vizualizāciju faili ir veiksmīgi saglabāti!")
    print("="*80)
    
def create_detailed_comparison_table(results_df):
    """Create a detailed comparison table"""
    
    # Create a formatted table for easy reading
    comparison_table = results_df.pivot_table(
        values=['accuracy', 'precision', 'recall', 'f1'], 
        index='model', 
        columns='condition',
        aggfunc='first'
    )
    
    # Round to 4 decimal places
    comparison_table = comparison_table.round(4)
    
    # Save to CSV
    comparison_table.to_csv('detailed_model_comparison_table.csv')
    print("Saved: detailed_model_comparison_table.csv")
    
    # Print formatted table
    print("\n" + "="*100)
    print(" DETALIZĒTA VEIKTSPĒJAS SALĪDZINĀJUMA TABULA")
    print("="*100)
    print(comparison_table.to_string())

def print_summary_analysis(results_df):
    """Print key insights from the analysis"""
    print("\n" + "="*80)
    print(" REZULTĀTU KOPSAVILKUMS")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 DATU KOPAS PĀRSKATS:")
    if not results_df.empty:
        sample_count = results_df.iloc[0]['sample_count']
        print(f"  • Kopējais testa paraugu skaits: {sample_count}")
        print(f"  • Testētie modeļi: {len(results_df['model'].unique())}")
        print(f"  • Testētie nosacījumi: {len(results_df['condition'].unique())}")
    
    trained_results = results_df[results_df['condition'] == 'trained']
    untrained_results = results_df[results_df['condition'] == 'untrained']
    
    if not trained_results.empty:
        print("\n🏆 LABĀKIE MODEĻI (APMĀCĪTI):")
        metric_names = {'accuracy': 'Precizitāte', 'f1': 'F1 rezultāts', 
                       'precision': 'Precīzums', 'recall': 'Jutība'}
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            best_model = trained_results.loc[trained_results[metric].idxmax()]
            print(f"  • {metric_names[metric]}: {best_model['model']} ({best_model[metric]:.4f})")
    
    if not untrained_results.empty:
        print("\n🏆 LABĀKIE MODEĻI (NEAPMĀCĪTI):")
        for metric in ['accuracy', 'f1']:
            best_model = untrained_results.loc[untrained_results[metric].idxmax()]
            print(f"  • {metric_names[metric]}: {best_model['model']} ({best_model[metric]:.4f})")
    
    # Training improvement analysis
    print("\n📈 APMĀCĪBAS UZLABOJUMU ANALĪZE:")
    improvements = []
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        if len(model_data) == 2:
            trained = model_data[model_data['condition'] == 'trained'].iloc[0]
            untrained = model_data[model_data['condition'] == 'untrained'].iloc[0]
            
            f1_improvement = trained['f1'] - untrained['f1']
            accuracy_improvement = trained['accuracy'] - untrained['accuracy']
            improvements.append(f1_improvement)
            
            print(f"  • {model}:")
            print(f"    - F1 rezultāts: {f1_improvement:+.4f}")
            print(f"    - Precizitāte: {accuracy_improvement:+.4f}")
    
    # Overall effectiveness
    if improvements:
        avg_improvement = np.mean(improvements)
        positive_improvements = sum(1 for x in improvements if x > 0)
        total_models = len(improvements)
        
        print(f"\n📊 KOPĒJĀ APMĀCĪBAS EFEKTIVITĀTE:")
        print(f"  • Vidējais F1 uzlabojums: {avg_improvement:+.4f}")
        print(f"  • Uzlabojušies modeļi: {positive_improvements}/{total_models}")
        
        if avg_improvement > 0.02:
            print(f"  ✅ Apmācības piemēri ir vispārēji efektīvi")
        elif avg_improvement > -0.02:
            print(f"  ⚠️  Apmācība uzrāda jauktos rezultātus")
        else:
            print(f"  ❌ Apmācība var būt pretproduktīva")
    
    # Local vs Cloud analysis
    local_results = results_df[results_df['deployment'] == 'Local']
    cloud_results = results_df[results_df['deployment'] == 'Cloud']
    
    if not local_results.empty and not cloud_results.empty:
        print(f"\n🏢 LOKĀLĀ un MĀKOŅA MODEĻU ANALĪZE:")
        
        # Compare best performers
        best_local_f1 = local_results['f1'].max()
        best_cloud_f1 = cloud_results['f1'].max()
        best_local_model = local_results.loc[local_results['f1'].idxmax(), 'model']
        best_cloud_model = cloud_results.loc[cloud_results['f1'].idxmax(), 'model']
        
        print(f"  • Labākais lokālais modelis: {best_local_model} (F1: {best_local_f1:.4f})")
        print(f"  • Labākais mākoņa modelis: {best_cloud_model} (F1: {best_cloud_f1:.4f})")
        
        if best_local_f1 > best_cloud_f1:
            print(f"  ✅ Lokālie modeļi var konkurēt ar mākoņa modeļiem")
        else:
            print(f"  📡 Mākoņa modeļi pārspēj lokālos modeļus")
            print(f"  💡 Apsveriet privātuma un veiktspējas kompromisus")
    
    # Model size insights (for local models)
    local_trained = results_df[(results_df['deployment'] == 'Local') & (results_df['condition'] == 'trained')]
    if len(local_trained) > 1:
        print(f"\n📏 MODEĻA IZMĒRA ANALĪZE (LOKĀLIE MODEĻI):")
        
        # Sort by model size
        size_order = {'1.7B': 1, '7B': 2, '16B': 3}
        local_trained_sorted = local_trained.copy()
        local_trained_sorted['size_order'] = local_trained_sorted['model_size'].map(size_order)
        local_trained_sorted = local_trained_sorted.sort_values('size_order')
        
        print("  • Veiktspēja pēc modeļa izmēra:")
        for _, row in local_trained_sorted.iterrows():
            print(f"    - {row['model']} ({row['model_size']}): F1 = {row['f1']:.4f}")
        
        # Check if larger models perform better
        f1_scores = local_trained_sorted['f1'].tolist()
        if len(f1_scores) >= 3:
            if f1_scores[-1] > f1_scores[0]:  # Largest vs smallest
                print(f"  📈 Lielāki modeļi parasti darbojas labāk")
            else:
                print(f"  🤔 Izmērs negarantē labāku veiktspēju")
    
    # Key recommendations
    print(f"\n💡 GALVENĀS ATZINĪBAS PROMOCIJAS DARBĀ:")
    print(f"  1. Salīdzināt apmācības efektivitāti dažādās modeļu arhitektūrās")
    print(f"  2. Analizēt privātuma un veiktspējas kompromisu starp lokālajiem un mākoņa modeļiem")
    print(f"  3. Pārbaudīt, vai modeļa izmērs korelē ar depresijas noteikšanas precizitāti")
    print(f"  4. Apsvērt viltus pozitīvu un viltus negatīvu rezultātu praktiskās sekas")
    print(f"  5. Apspriest neobjektivitāti depresijas klasifikācijā lielākajā daļā modeļu")

if __name__ == "__main__":
    print("Sāk visaptverošu visu modeļu rezultātu analīzi...")
    print("Meklēju failus:")
    print("- deepseek-v2_16b_trained_results.csv")
    print("- deepseek-v2_16b_untrained_results.csv")
    print("- medllama2_7b_trained_results.csv")
    print("- medllama2_7b_untrained_results.csv")
    print("- qwen3_1.7b_trained_results.csv")
    print("- qwen3_1.7b_untrained_results.csv")
    print("- chatgpt_4o_mini_trained_results.csv")
    print("- chatgpt_4o_mini_untrained_results.csv")
    print("- ai_comparison_test/comparison_test_5431_reference.csv")
    print("\n" + "="*50)
    
    results_df = analyze_all_results()
    
    if results_df is not None:
        print(f"\n✅ Analīze pabeigta!")
        print(f"📊 Rezultāti saglabāti 'comprehensive_model_comparison.csv'")
        print(f"📈 Vizualizācijas saglabātas kā PNG faili")
        print(f"📋 Detalizēta salīdzināšanas tabula saglabāta failā 'detailed_model_comparison_table.csv'")
        print(f"📋 Skatiet iepriekš minēto kopsavilkumu, lai iegūtu galveno informāciju")
    else:
        print(f"\n❌ Analīze neizdevās — pārbaudiet failu ceļus un formātus")