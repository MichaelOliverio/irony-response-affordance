"""
Script per generare statistiche sui dataset filtrati
"""

import json
import os
from pathlib import Path
from collections import Counter
import statistics
import spacy

# Carica il modello spaCy per l'inglese (leggero per efficienza)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  Modello spaCy 'en_core_web_sm' non trovato.")
    print("    Installalo con: python -m spacy download en_core_web_sm")
    nlp = None


def load_dataset(file_path):
    """Carica il dataset JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_tokens(text):
    """Conta i token usando il tokenizer di spaCy"""
    if not text:
        return 0
    
    if nlp is None:
        # Fallback a splitting semplice se spaCy non è disponibile
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return len(tokens)
    
    # Usa solo il tokenizer di spaCy (disabilita altre pipeline per velocità)
    doc = nlp(text, disable=["parser", "ner", "lemmatizer", "textcat"])
    return len(doc)



def analyze_dialogue_structure(entries):
    """Analizza la struttura dei dialoghi e conta i token"""
    total_turns = []
    total_utterances = []
    total_tokens = []
    
    for entry in entries:
        dialogue = entry.get('dialogue', {})
        turns = dialogue.get('turns', [])
        total_turns.append(len(turns))
        
        # Conta le utterances totali e i token
        # Ogni turn contiene 'u1' e 'u2' come utterances
        utterances = 0
        dialogue_tokens = 0
        for turn in turns:
            # u1 e u2 sono stringhe, non liste
            u1_text = turn.get('u1', '')
            u2_text = turn.get('u2', '')
            
            # Conta le utterances (u1 e u2 per ogni turn)
            if u1_text:
                utterances += 1
                dialogue_tokens += count_tokens(u1_text)
            if u2_text:
                utterances += 1
                dialogue_tokens += count_tokens(u2_text)
        
        total_utterances.append(utterances)
        total_tokens.append(dialogue_tokens)
    
    return {
        'total_dialogues': len(entries),
        'turns': {
            'mean': statistics.mean(total_turns) if total_turns else 0,
            'median': statistics.median(total_turns) if total_turns else 0,
            'min': min(total_turns) if total_turns else 0,
            'max': max(total_turns) if total_turns else 0,
            'stdev': statistics.stdev(total_turns) if len(total_turns) > 1 else 0
        },
        'utterances_per_dialogue': {
            'mean': statistics.mean(total_utterances) if total_utterances else 0,
            'median': statistics.median(total_utterances) if total_utterances else 0,
            'min': min(total_utterances) if total_utterances else 0,
            'max': max(total_utterances) if total_utterances else 0,
            'stdev': statistics.stdev(total_utterances) if len(total_utterances) > 1 else 0
        },
        'tokens_per_dialogue': {
            'mean': statistics.mean(total_tokens) if total_tokens else 0,
            'median': statistics.median(total_tokens) if total_tokens else 0,
            'min': min(total_tokens) if total_tokens else 0,
            'max': max(total_tokens) if total_tokens else 0,
            'stdev': statistics.stdev(total_tokens) if len(total_tokens) > 1 else 0,
            'total': sum(total_tokens)
        }
    }


def analyze_annotations(entries, dataset_name):
    """Analizza le annotazioni degli observer e speaker"""
    
    # Determina se è CSC o MultiPICo
    is_csc = 'CSC' in dataset_name.upper()
    label_key = 'sarcasm' if is_csc else 'irony'
    
    # Statistiche generali
    total_records = len(entries)
    records_with_speaker = 0
    observer_counts = []
    
    # Distribuzione labels complessiva
    aggregate_labels = []
    average_scores = []
    
    # Distribuzione per speaker
    speaker_labels = []
    speaker_scores = []
    
    # Distribuzione per observer
    observer_labels = []
    observer_scores = []
    
    for entry in entries:
        annotations = entry.get('annotations', {})
        
        # Annotazione speaker
        speaker_annotation = annotations.get('speaker_annotation')
        if speaker_annotation:
            records_with_speaker += 1
            speaker_label = speaker_annotation.get(f'{label_key}_label')
            speaker_score = speaker_annotation.get(f'{label_key}_score')
            
            if speaker_label is not None:
                speaker_labels.append(speaker_label)
            if speaker_score is not None:
                speaker_scores.append(speaker_score)
        
        # Annotazioni observer
        observers = annotations.get('observers_annotations', [])
        observer_counts.append(len(observers))
        
        for obs in observers:
            obs_label = obs.get(f'{label_key}_label')
            obs_score = obs.get(f'{label_key}_score')
            
            if obs_label is not None:
                observer_labels.append(obs_label)
            if obs_score is not None:
                observer_scores.append(obs_score)
        
        # Label aggregata
        agg_label = annotations.get(f'aggregate_{label_key}_label')
        if agg_label is not None:
            aggregate_labels.append(agg_label)
        
        # Score medio
        avg_score = annotations.get(f'average_{label_key}_score')
        if avg_score is not None:
            average_scores.append(avg_score)
    
    # Calcola statistiche
    stats = {
        'total_records': total_records,
        'records_with_speaker_annotation': records_with_speaker,
        'percentage_with_speaker': (records_with_speaker / total_records * 100) if total_records > 0 else 0,
        'observers_per_record': {
            'mean': statistics.mean(observer_counts) if observer_counts else 0,
            'median': statistics.median(observer_counts) if observer_counts else 0,
            'min': min(observer_counts) if observer_counts else 0,
            'max': max(observer_counts) if observer_counts else 0,
            'stdev': statistics.stdev(observer_counts) if len(observer_counts) > 1 else 0
        },
        'label_type': label_key
    }
    
    # Distribuzione complessiva (aggregate)
    if aggregate_labels:
        label_counter = Counter(aggregate_labels)
        total = len(aggregate_labels)
        stats['aggregate_distribution'] = {
            f'{label_key}': label_counter.get(1, 0),
            f'non_{label_key}': label_counter.get(0, 0),
            'percentage': {
                f'{label_key}': (label_counter.get(1, 0) / total * 100) if total > 0 else 0,
                f'non_{label_key}': (label_counter.get(0, 0) / total * 100) if total > 0 else 0
            }
        }
    
    # Distribuzione speaker
    if speaker_labels:
        label_counter = Counter(speaker_labels)
        total = len(speaker_labels)
        stats['speaker_distribution'] = {
            f'{label_key}': label_counter.get(1, 0),
            f'non_{label_key}': label_counter.get(0, 0),
            'total_annotations': total,
            'percentage': {
                f'{label_key}': (label_counter.get(1, 0) / total * 100) if total > 0 else 0,
                f'non_{label_key}': (label_counter.get(0, 0) / total * 100) if total > 0 else 0
            }
        }
    
    if speaker_scores:
        stats['speaker_scores'] = {
            'mean': statistics.mean(speaker_scores),
            'median': statistics.median(speaker_scores),
            'min': min(speaker_scores),
            'max': max(speaker_scores),
            'stdev': statistics.stdev(speaker_scores) if len(speaker_scores) > 1 else 0
        }
    
    # Distribuzione observer
    if observer_labels:
        label_counter = Counter(observer_labels)
        total = len(observer_labels)
        stats['observer_distribution'] = {
            f'{label_key}': label_counter.get(1, 0),
            f'non_{label_key}': label_counter.get(0, 0),
            'total_annotations': total,
            'percentage': {
                f'{label_key}': (label_counter.get(1, 0) / total * 100) if total > 0 else 0,
                f'non_{label_key}': (label_counter.get(0, 0) / total * 100) if total > 0 else 0
            }
        }
    
    if observer_scores:
        stats['observer_scores'] = {
            'mean': statistics.mean(observer_scores),
            'median': statistics.median(observer_scores),
            'min': min(observer_scores),
            'max': max(observer_scores),
            'stdev': statistics.stdev(observer_scores) if len(observer_scores) > 1 else 0
        }
    
    # Score medio complessivo
    if average_scores:
        stats['average_scores_overall'] = {
            'mean': statistics.mean(average_scores),
            'median': statistics.median(average_scores),
            'min': min(average_scores),
            'max': max(average_scores),
            'stdev': statistics.stdev(average_scores) if len(average_scores) > 1 else 0
        }
    
    return stats

def print_statistics(dataset_name, stats):
    """Stampa le statistiche in modo formattato"""
    print(f"\n{'='*80}")
    print(f"STATISTICHE DATASET: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Metadata
    if 'metadata' in stats:
        meta = stats['metadata']
        print("📊 METADATA GENERALE")
        print(f"  Dataset originale: {meta.get('dataset_name', 'N/A')}")
        print(f"  Modello Judge: {meta.get('judge_model', 'N/A')}")
        print(f"  Entries totali: {meta.get('total_entries', 'N/A')}")
        print(f"  Entries filtrate: {meta.get('filtered_entries_count', 'N/A')}")
        print(f"  Entries rimosse: {meta.get('removed_entries_count', 'N/A')}")
        print(f"  Soglia filtraggio: {meta.get('threshold', 'N/A')}")
        print(f"  Score medio (tutti): {meta.get('average_score_all', 0):.4f}")
        print(f"  Score medio (filtrati): {meta.get('average_score_filtered', 0):.4f}")
        print()
    
    # Struttura dialoghi
    if 'dialogue_structure' in stats:
        ds = stats['dialogue_structure']
        print("💬 STRUTTURA DIALOGHI")
        print(f"  Dialoghi totali: {ds['total_dialogues']}")
        print(f"  Turni per dialogo:")
        print(f"    Media: {ds['turns']['mean']:.2f}")
        print(f"    Mediana: {ds['turns']['median']:.2f}")
        print(f"    Min-Max: {ds['turns']['min']}-{ds['turns']['max']}")
        print(f"    Dev. Standard: {ds['turns']['stdev']:.2f}")
        print(f"  Utterances per dialogo:")
        print(f"    Media: {ds['utterances_per_dialogue']['mean']:.2f}")
        print(f"    Mediana: {ds['utterances_per_dialogue']['median']:.2f}")
        print(f"    Min-Max: {ds['utterances_per_dialogue']['min']}-{ds['utterances_per_dialogue']['max']}")
        print(f"    Dev. Standard: {ds['utterances_per_dialogue']['stdev']:.2f}")
        print(f"  Token per dialogo:")
        print(f"    Media: {ds['tokens_per_dialogue']['mean']:.2f}")
        print(f"    Mediana: {ds['tokens_per_dialogue']['median']:.2f}")
        print(f"    Min-Max: {ds['tokens_per_dialogue']['min']}-{ds['tokens_per_dialogue']['max']}")
        print(f"    Dev. Standard: {ds['tokens_per_dialogue']['stdev']:.2f}")
        print(f"    Totale: {ds['tokens_per_dialogue']['total']}")
        print()
    
    # Annotazioni originali
    if 'annotations' in stats:
        ann = stats['annotations']
        label_type = ann.get('label_type', 'label')
        
        print(f"📝 ANNOTAZIONI ORIGINALI ({label_type.upper()})")
        print(f"  Record totali: {ann['total_records']}")
        print(f"  Record con annotazione speaker: {ann['records_with_speaker_annotation']} ({ann['percentage_with_speaker']:.2f}%)")
        print(f"  Observer per record:")
        print(f"    Media: {ann['observers_per_record']['mean']:.2f}")
        print(f"    Mediana: {ann['observers_per_record']['median']:.2f}")
        print(f"    Min-Max: {ann['observers_per_record']['min']}-{ann['observers_per_record']['max']}")
        print(f"    Dev. Standard: {ann['observers_per_record']['stdev']:.2f}")
        print()
        
        # Distribuzione aggregata
        if 'aggregate_distribution' in ann:
            agg = ann['aggregate_distribution']
            print(f"  Distribuzione AGGREGATA:")
            print(f"    {label_type.capitalize()}: {agg[label_type]} ({agg['percentage'][label_type]:.2f}%)")
            print(f"    Non-{label_type}: {agg[f'non_{label_type}']} ({agg['percentage'][f'non_{label_type}']:.2f}%)")
            print()
        
        # Distribuzione speaker
        if 'speaker_distribution' in ann:
            spk = ann['speaker_distribution']
            print(f"  Distribuzione SPEAKER:")
            print(f"    Annotazioni totali: {spk['total_annotations']}")
            print(f"    {label_type.capitalize()}: {spk[label_type]} ({spk['percentage'][label_type]:.2f}%)")
            print(f"    Non-{label_type}: {spk[f'non_{label_type}']} ({spk['percentage'][f'non_{label_type}']:.2f}%)")
            
            if 'speaker_scores' in ann:
                ss = ann['speaker_scores']
                print(f"    Score - Media: {ss['mean']:.4f}, Mediana: {ss['median']:.4f}, Min-Max: {ss['min']:.4f}-{ss['max']:.4f}")
            print()
        
        # Distribuzione observer
        if 'observer_distribution' in ann:
            obs = ann['observer_distribution']
            print(f"  Distribuzione OBSERVER:")
            print(f"    Annotazioni totali: {obs['total_annotations']}")
            print(f"    {label_type.capitalize()}: {obs[label_type]} ({obs['percentage'][label_type]:.2f}%)")
            print(f"    Non-{label_type}: {obs[f'non_{label_type}']} ({obs['percentage'][f'non_{label_type}']:.2f}%)")
            
            if 'observer_scores' in ann:
                os = ann['observer_scores']
                print(f"    Score - Media: {os['mean']:.4f}, Mediana: {os['median']:.4f}, Min-Max: {os['min']:.4f}-{os['max']:.4f}")
            print()
        
        # Score medio complessivo
        if 'average_scores_overall' in ann:
            avg = ann['average_scores_overall']
            print(f"  Score medio COMPLESSIVO:")
            print(f"    Media: {avg['mean']:.4f}")
            print(f"    Mediana: {avg['median']:.4f}")
            print(f"    Min-Max: {avg['min']:.4f}-{avg['max']:.4f}")
            print(f"    Dev. Standard: {avg['stdev']:.4f}")
            print()



def combine_statistics(all_stats_list):
    """Combina le statistiche di più dataset"""
    combined = {
        'total_datasets': len(all_stats_list),
        'datasets_names': [s['dataset_name'] for s in all_stats_list]
    }
    
    # Somma totali
    total_dialogues = sum(s['stats']['dialogue_structure']['total_dialogues'] for s in all_stats_list)
    total_records = sum(s['stats']['annotations']['total_records'] for s in all_stats_list)
    total_tokens = sum(s['stats']['dialogue_structure']['tokens_per_dialogue']['total'] for s in all_stats_list)
    
    combined['totals'] = {
        'total_dialogues': total_dialogues,
        'total_records': total_records,
        'total_tokens': total_tokens
    }
    
    # Media ponderata dei token per dialogo
    all_tokens = []
    all_turns = []
    all_utterances = []
    all_observer_counts = []
    
    for s in all_stats_list:
        # Ricrea le liste originali usando le statistiche
        ds = s['stats']['dialogue_structure']
        ann = s['stats']['annotations']
        
        # Estrai informazioni
        num_dialogues = ds['total_dialogues']
        avg_tokens = ds['tokens_per_dialogue']['mean']
        avg_turns = ds['turns']['mean']
        avg_utterances = ds['utterances_per_dialogue']['mean']
        avg_observers = ann['observers_per_record']['mean']
        
        # Approssima i valori totali (non perfetto ma buona stima)
        all_tokens.extend([avg_tokens] * num_dialogues)
        all_turns.extend([avg_turns] * num_dialogues)
        all_utterances.extend([avg_utterances] * num_dialogues)
        all_observer_counts.extend([avg_observers] * num_dialogues)
    
    combined['dialogue_structure'] = {
        'total_dialogues': total_dialogues,
        'turns_per_dialogue': {
            'mean': statistics.mean(all_turns) if all_turns else 0,
            'median': statistics.median(all_turns) if all_turns else 0
        },
        'utterances_per_dialogue': {
            'mean': statistics.mean(all_utterances) if all_utterances else 0,
            'median': statistics.median(all_utterances) if all_utterances else 0
        },
        'tokens_per_dialogue': {
            'mean': statistics.mean(all_tokens) if all_tokens else 0,
            'median': statistics.median(all_tokens) if all_tokens else 0,
            'total': total_tokens
        }
    }
    
    # Combina annotazioni
    total_with_speaker = sum(s['stats']['annotations']['records_with_speaker_annotation'] for s in all_stats_list)
    
    combined['annotations'] = {
        'total_records': total_records,
        'records_with_speaker_annotation': total_with_speaker,
        'percentage_with_speaker': (total_with_speaker / total_records * 100) if total_records > 0 else 0,
        'observers_per_record': {
            'mean': statistics.mean(all_observer_counts) if all_observer_counts else 0,
            'median': statistics.median(all_observer_counts) if all_observer_counts else 0
        }
    }
    
    # Combina distribuzioni unificando sarcasm e irony
    # Usiamo "figurative" come label unificata per sarcasm + irony
    combined['unified_distribution'] = {
        'aggregate': {'figurative': 0, 'non_figurative': 0},
        'speaker': {'figurative': 0, 'non_figurative': 0},
        'observer': {'figurative': 0, 'non_figurative': 0}
    }
    
    # Mantieni anche le distribuzioni separate per riferimento
    combined['separate_distributions'] = {}
    
    for dataset_stats in all_stats_list:
        ann = dataset_stats['stats']['annotations']
        label_type = ann.get('label_type', 'label')
        
        # Salva distribuzione separata
        if label_type not in combined['separate_distributions']:
            combined['separate_distributions'][label_type] = {
                'aggregate': {label_type: 0, f'non_{label_type}': 0},
                'speaker': {label_type: 0, f'non_{label_type}': 0},
                'observer': {label_type: 0, f'non_{label_type}': 0}
            }
        
        # Aggrega nelle distribuzioni separate
        if 'aggregate_distribution' in ann:
            agg = ann['aggregate_distribution']
            combined['separate_distributions'][label_type]['aggregate'][label_type] += agg.get(label_type, 0)
            combined['separate_distributions'][label_type]['aggregate'][f'non_{label_type}'] += agg.get(f'non_{label_type}', 0)
            
            # Aggrega anche nella distribuzione unificata
            combined['unified_distribution']['aggregate']['figurative'] += agg.get(label_type, 0)
            combined['unified_distribution']['aggregate']['non_figurative'] += agg.get(f'non_{label_type}', 0)
        
        # Speaker
        if 'speaker_distribution' in ann:
            spk = ann['speaker_distribution']
            combined['separate_distributions'][label_type]['speaker'][label_type] += spk.get(label_type, 0)
            combined['separate_distributions'][label_type]['speaker'][f'non_{label_type}'] += spk.get(f'non_{label_type}', 0)
            
            # Aggrega anche nella distribuzione unificata
            combined['unified_distribution']['speaker']['figurative'] += spk.get(label_type, 0)
            combined['unified_distribution']['speaker']['non_figurative'] += spk.get(f'non_{label_type}', 0)
        
        # Observer
        if 'observer_distribution' in ann:
            obs = ann['observer_distribution']
            combined['separate_distributions'][label_type]['observer'][label_type] += obs.get(label_type, 0)
            combined['separate_distributions'][label_type]['observer'][f'non_{label_type}'] += obs.get(f'non_{label_type}', 0)
            
            # Aggrega anche nella distribuzione unificata
            combined['unified_distribution']['observer']['figurative'] += obs.get(label_type, 0)
            combined['unified_distribution']['observer']['non_figurative'] += obs.get(f'non_{label_type}', 0)
    
    # Calcola percentuali per la distribuzione unificata
    for dist_type in ['aggregate', 'speaker', 'observer']:
        dist = combined['unified_distribution'][dist_type]
        total = dist['figurative'] + dist['non_figurative']
        
        dist['percentage'] = {
            'figurative': (dist['figurative'] / total * 100) if total > 0 else 0,
            'non_figurative': (dist['non_figurative'] / total * 100) if total > 0 else 0
        }
        dist['total'] = total
    
    # Calcola percentuali per le distribuzioni separate
    for label_type, label_data in combined['separate_distributions'].items():
        for dist_type in ['aggregate', 'speaker', 'observer']:
            dist = label_data[dist_type]
            total = dist.get(label_type, 0) + dist.get(f'non_{label_type}', 0)
            
            dist['percentage'] = {
                label_type: (dist.get(label_type, 0) / total * 100) if total > 0 else 0,
                f'non_{label_type}': (dist.get(f'non_{label_type}', 0) / total * 100) if total > 0 else 0
            }
            dist['total'] = total
    
    return combined


def print_combined_statistics(combined):
    """Stampa le statistiche combinate"""
    print(f"\n{'='*80}")
    print(f"STATISTICHE TOTALI COMBINATE")
    print(f"{'='*80}\n")
    
    print(f"📚 DATASET COMBINATI: {', '.join(combined['datasets_names'])}")
    print(f"  Numero di dataset: {combined['total_datasets']}")
    print()
    
    # Totali
    print("📊 TOTALI")
    print(f"  Dialoghi totali: {combined['totals']['total_dialogues']}")
    print(f"  Record totali: {combined['totals']['total_records']}")
    print(f"  Token totali: {combined['totals']['total_tokens']}")
    print()
    
    # Struttura dialoghi
    if 'dialogue_structure' in combined:
        ds = combined['dialogue_structure']
        print("💬 STRUTTURA DIALOGHI (media)")
        print(f"  Turni per dialogo: {ds['turns_per_dialogue']['mean']:.2f} (mediana: {ds['turns_per_dialogue']['median']:.2f})")
        print(f"  Utterances per dialogo: {ds['utterances_per_dialogue']['mean']:.2f} (mediana: {ds['utterances_per_dialogue']['median']:.2f})")
        print(f"  Token per dialogo: {ds['tokens_per_dialogue']['mean']:.2f} (mediana: {ds['tokens_per_dialogue']['median']:.2f})")
        print()
    
    # Annotazioni
    if 'annotations' in combined:
        ann = combined['annotations']
        print("📝 ANNOTAZIONI")
        print(f"  Record con speaker annotation: {ann['records_with_speaker_annotation']} ({ann['percentage_with_speaker']:.2f}%)")
        print(f"  Observer per record (media): {ann['observers_per_record']['mean']:.2f}")
        print()
    
    # Distribuzione UNIFICATA (sarcasm + irony = figurative)
    if 'unified_distribution' in combined:
        print("🏷️  DISTRIBUZIONE UNIFICATA (Sarcasm + Irony)")
        print()
        
        unified = combined['unified_distribution']
        
        # Aggregata
        if 'aggregate' in unified:
            agg = unified['aggregate']
            print(f"  AGGREGATA (totale: {agg.get('total', 0)}):")
            print(f"    Figurativo (sarcasm/irony): {agg.get('figurative', 0)} ({agg.get('percentage', {}).get('figurative', 0):.2f}%)")
            print(f"    Non-figurativo: {agg.get('non_figurative', 0)} ({agg.get('percentage', {}).get('non_figurative', 0):.2f}%)")
            print()
        
        # Speaker
        if 'speaker' in unified:
            spk = unified['speaker']
            print(f"  SPEAKER (totale: {spk.get('total', 0)}):")
            print(f"    Figurativo (sarcasm/irony): {spk.get('figurative', 0)} ({spk.get('percentage', {}).get('figurative', 0):.2f}%)")
            print(f"    Non-figurativo: {spk.get('non_figurative', 0)} ({spk.get('percentage', {}).get('non_figurative', 0):.2f}%)")
            print()
        
        # Observer
        if 'observer' in unified:
            obs = unified['observer']
            print(f"  OBSERVER (totale: {obs.get('total', 0)}):")
            print(f"    Figurativo (sarcasm/irony): {obs.get('figurative', 0)} ({obs.get('percentage', {}).get('figurative', 0):.2f}%)")
            print(f"    Non-figurativo: {obs.get('non_figurative', 0)} ({obs.get('percentage', {}).get('non_figurative', 0):.2f}%)")
            print()
    
    # Distribuzioni SEPARATE per riferimento
    if 'separate_distributions' in combined:
        print("📋 DISTRIBUZIONI SEPARATE (per riferimento)")
        print()
        
        for label_type, dist_data in combined['separate_distributions'].items():
            print(f"  {label_type.upper()}")
            
            # Aggregata
            if 'aggregate' in dist_data:
                agg = dist_data['aggregate']
                print(f"    AGGREGATA (totale: {agg.get('total', 0)}):")
                print(f"      {label_type.capitalize()}: {agg.get(label_type, 0)} ({agg.get('percentage', {}).get(label_type, 0):.2f}%)")
                print(f"      Non-{label_type}: {agg.get(f'non_{label_type}', 0)} ({agg.get('percentage', {}).get(f'non_{label_type}', 0):.2f}%)")
            
            # Speaker
            if 'speaker' in dist_data:
                spk = dist_data['speaker']
                print(f"    SPEAKER (totale: {spk.get('total', 0)}):")
                print(f"      {label_type.capitalize()}: {spk.get(label_type, 0)} ({spk.get('percentage', {}).get(label_type, 0):.2f}%)")
                print(f"      Non-{label_type}: {spk.get(f'non_{label_type}', 0)} ({spk.get('percentage', {}).get(f'non_{label_type}', 0):.2f}%)")
            
            # Observer
            if 'observer' in dist_data:
                obs = dist_data['observer']
                print(f"    OBSERVER (totale: {obs.get('total', 0)}):")
                print(f"      {label_type.capitalize()}: {obs.get(label_type, 0)} ({obs.get('percentage', {}).get(label_type, 0):.2f}%)")
                print(f"      Non-{label_type}: {obs.get(f'non_{label_type}', 0)} ({obs.get('percentage', {}).get(f'non_{label_type}', 0):.2f}%)")
            
            print()


def save_statistics_to_json(stats, output_path):
    """Salva le statistiche in un file JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✅ Statistiche salvate in: {output_path}")


def main():
    # Percorsi dataset
    datasets_dir = Path(r"c:\Users\micha\OneDrive\Desktop\Università\Progetti\appropriateness-of-irony\3_experiments\datasets")
    
    datasets = [
        "CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_final.json",
        "MultiPICo_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_final.json"
    ]
    
    all_stats = []
    
    for dataset_file in datasets:
        dataset_path = datasets_dir / dataset_file
        
        if not dataset_path.exists():
            print(f"⚠️  File non trovato: {dataset_path}")
            continue
        
        # Carica dataset
        print(f"\n🔄 Caricamento dataset: {dataset_file}...")
        data = load_dataset(dataset_path)
        entries = data.get('entries', [])
        metadata = data.get('metadata', {})
        
        # Nome dataset per distinguere CSC da MultiPICo
        dataset_name = dataset_file.replace('_final.json', '')
        
        # Calcola statistiche
        print(f"📊 Calcolo statistiche...")
        stats = {
            'metadata': metadata,
            'dialogue_structure': analyze_dialogue_structure(entries),
            'annotations': analyze_annotations(entries, dataset_name),
        }
        
        # Salva per statistiche combinate
        all_stats.append({
            'dataset_name': dataset_name,
            'stats': stats
        })
        
        # Stampa statistiche
        print_statistics(dataset_name, stats)
        
        # Salva statistiche
        output_path = datasets_dir / f"statistics_{dataset_file}"
        save_statistics_to_json(stats, output_path)
        
        print(f"\n{'-'*80}\n")
    
    # Calcola e stampa statistiche combinate
    if len(all_stats) > 1:
        print("\n" + "="*80)
        print("GENERAZIONE STATISTICHE COMBINATE")
        print("="*80)
        
        combined = combine_statistics(all_stats)
        print_combined_statistics(combined)
        
        # Salva statistiche combinate
        output_path = datasets_dir / "statistics_combined.json"
        save_statistics_to_json(combined, output_path)


if __name__ == "__main__":
    main()
