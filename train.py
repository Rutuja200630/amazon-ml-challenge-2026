import sys
import os
import time
import json
import argparse
import collections
import numpy as np

# Ensure code/business_entity_resolution/src is in python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import normalization as norm
from features import extract_features_for_pair, FEATURE_NAMES
from model import EntityMatcherModel
from evaluation import evaluate_predictions
from thresholding import apply_threshold_and_deduplication
from blocking import get_blocking_keys


def auto_detect_train_dir():
    candidates = ['dataset/train', 'student_resource/dataset/train']
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'train_ground_truth.tsv')):
            return c
    if os.path.isdir('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'train_ground_truth.tsv' in files:
                return root
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description='Train Business Entity Resolution Model with Hard Negative Mining')
    parser.add_argument('--train-dir', default=auto_detect_train_dir(),
                        help='Path to train dataset directory')
    parser.add_argument('--val-ids', default='experiments/val_s1_ids.txt',
                        help='Path to file containing held-out validation Source 1 IDs')
    parser.add_argument('--n-val', type=int, default=10000,
                        help='Number of validation Source 1 entities to evaluate')
    parser.add_argument('--n-train', type=int, default=150000,
                        help='Number of training Source 1 entities to sample')
    parser.add_argument('--top-k', type=int, default=200,
                        help='Number of candidates per S1 entity')
    parser.add_argument('--target-metric', default='macro_f2', choices=['macro_f2', 'macro_f1', 'macro_f05'],
                        help='Metric to optimize thresholding grid search for')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible training sampling')
    parser.add_argument('--n-distractors', type=int, default=300000,
                        help='Number of realistic background distractors to load from target sources')
    parser.add_argument('--model-type', default='xgboost_gpu', choices=['xgboost_gpu', 'xgboost', 'lightgbm'],
                        help='Model architecture to train (xgboost_gpu uses NVIDIA CUDA GPU)')
    parser.add_argument('--n-estimators', type=int, default=800,
                        help='Number of gradient boosted trees')
    parser.add_argument('--model-out', default='models/final_entity_matcher.joblib',
                        help='Output path for trained model')
    parser.add_argument('--meta-out', default='models/model_metadata.json',
                        help='Output path for model metadata')
    args = parser.parse_args()

    print('=== Training Business Entity Resolution Model with Hard Negatives ===')
    print(f'Train directory : {args.train_dir}')
    print(f'Validation split: {args.val_ids} (using {args.n_val:,} entities)')
    print(f'Training sample : {args.n_train:,} S1 entities')

    gt_file = os.path.join(args.train_dir, 'train_ground_truth.tsv')
    s1_file = os.path.join(args.train_dir, 'train_source1.tsv')
    s2_file = os.path.join(args.train_dir, 'train_source2.tsv')
    s3_file = os.path.join(args.train_dir, 'train_source3.tsv')

    if not os.path.isfile(gt_file):
        print(f'Error: Ground truth file not found at {gt_file}')
        sys.exit(1)

    # 1. Load Validation S1 set
    val_s1_ids = []
    if os.path.isfile(args.val_ids):
        with open(args.val_ids, 'r', encoding='utf-8') as f:
            for line in f:
                sid = line.strip()
                if sid:
                    val_s1_ids.append(sid)
                if len(val_s1_ids) >= args.n_val:
                    break
    val_s1_set = set(val_s1_ids)
    print(f'Loaded {len(val_s1_set):,} validation entities.')

    val_gt = {}
    val_true_targets = set()
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in val_s1_set:
                mids = p[1].split(',') if len(p) > 1 and p[1] else []
                val_gt[p[0]] = set(mids)
                val_true_targets.update(mids)

    # 2. Select Training S1 records (strictly disjoint from validation)
    all_train_s1_ids = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            sid = p[0]
            if sid not in val_s1_set:
                all_train_s1_ids.append(sid)

    rng = np.random.RandomState(args.seed)
    if len(all_train_s1_ids) > args.n_train:
        indices = rng.choice(len(all_train_s1_ids), size=args.n_train, replace=False)
        train_s1_ids = [all_train_s1_ids[i] for i in indices]
    else:
        train_s1_ids = all_train_s1_ids

    train_s1_set = set(train_s1_ids)
    train_gt = {}
    train_true_targets = set()
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in train_s1_set:
                mids = p[1].split(',') if len(p) > 1 and p[1] else []
                train_gt[p[0]] = set(mids)
                train_true_targets.update(mids)

    print(f'Training pool: {len(train_s1_ids):,} S1 records, {sum(len(v) for v in train_gt.values()):,} true positive links.')

    # 3. Load S1 raw records for train and val
    all_needed_s1 = val_s1_set | train_s1_set
    s1_raw = {}
    with open(s1_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in all_needed_s1:
                s1_raw[p[0]] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')

    s1_preprocessed = {}
    s1_blocking_keys = {}
    for sid, (rname, raddr, rcountry) in s1_raw.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, pnum, _ = norm.normalize_address(raddr)
        s1_preprocessed[sid] = (cn, core_n, ca, nums, pnum, rcountry)
        s1_blocking_keys[sid] = get_blocking_keys(rname, raddr, rcountry)

    # 4. Load Target records (True targets + realistic background distractors)
    all_needed_targets = val_true_targets | train_true_targets
    target_raw = {}
    max_d_per_src = args.n_distractors // 2
    d_s2 = 0
    with open(s2_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            tid = p[0]
            if tid in all_needed_targets or d_s2 < max_d_per_src:
                target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
                if tid not in all_needed_targets:
                    d_s2 += 1

    d_s3 = 0
    with open(s3_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            tid = p[0]
            if tid in all_needed_targets or d_s3 < max_d_per_src:
                target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
                if tid not in all_needed_targets:
                    d_s3 += 1

    print(f'Total target pool: {len(target_raw):,} records loaded (including {d_s2 + d_s3:,} distractors).')

    target_preprocessed = {}
    target_keys = {}
    for tid, (rname, raddr, rcountry) in target_raw.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, pnum, _ = norm.normalize_address(raddr)
        target_preprocessed[tid] = (cn, core_n, ca, nums, pnum, rcountry)
        target_keys[tid] = get_blocking_keys(rname, raddr, rcountry)

    # Inverted index by country
    print('Building country inverted indices...')
    indices = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, tkeys in target_keys.items():
        rcountry = target_preprocessed[tid][5]
        for k in tkeys:
            indices[rcountry][k].append(tid)

    for c in indices:
        for k in list(indices[c].keys()):
            limit = 4000 if (k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact') or k[0].startswith('gram') or k[0].startswith('snd')) else 1800
            if len(indices[c][k]) > limit:
                del indices[c][k]

    def get_candidates(sid, top_k=200):
        s_keys = s1_blocking_keys[sid]
        country = s1_preprocessed[sid][5]
        c_index = indices[country]
        counts = collections.Counter()
        for k in s_keys:
            if k in c_index:
                counts.update(c_index[k])
        if counts:
            return counts.most_common(top_k)
        return []

    # 5. Build Training Feature Matrix with Active Hard Negative Mining
    print(f'Extracting features for training pairs using multi-core CPU ({os.cpu_count() or 4} workers)...')
    t_feat_start = time.time()
    train_target_set = set(target_preprocessed.keys())

    import concurrent.futures

    def process_s1_train_entity(sid):
        true_mids = train_gt.get(sid, set()) & train_target_set
        cands = get_candidates(sid, top_k=args.top_k)
        cand_mids = {tid: count for tid, count in cands}
        s1_tup = s1_preprocessed[sid][:5]
        local_X = []
        local_y = []

        for mid in true_mids:
            if mid in target_preprocessed:
                t_tup = target_preprocessed[mid][:5]
                sh = cand_mids.get(mid, 1)
                feats = extract_features_for_pair(s1_tup, t_tup, mid, sh)
                local_X.append(feats)
                local_y.append(1)

        neg_count = 0
        max_negs = max(10, len(true_mids) * 10)
        for tid, sh in cands:
            if tid not in true_mids and tid in target_preprocessed:
                t_tup = target_preprocessed[tid][:5]
                feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
                local_X.append(feats)
                local_y.append(0)
                neg_count += 1
                if neg_count >= max_negs:
                    break
        return local_X, local_y

    num_workers = min(16, max(2, (os.cpu_count() or 4)))
    X_train_list = []
    y_train_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(process_s1_train_entity, train_s1_ids, chunksize=500)
        for res_X, res_y in futures:
            if res_X:
                X_train_list.extend(res_X)
                y_train_list.extend(res_y)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    print(f'Training dataset: X_train shape = {X_train.shape} (Positives = {pos_count:,}, Negatives = {neg_count:,}) in {time.time()-t_feat_start:.1f}s.')

    # 6. Fit Production Model (GPU Accelerated if xgboost_gpu)
    print(f'Training production {args.model_type} model on {"NVIDIA GPU (CUDA)" if "gpu" in args.model_type else "CPU"}...')
    t_train_start = time.time()
    final_model = EntityMatcherModel(
        args.model_type,
        n_estimators=args.n_estimators,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85
    )
    final_model.fit(X_train, y_train)
    print(f'Model trained in {time.time()-t_train_start:.1f}s.')

    # 7. Evaluate on Large Held-out Validation Set
    print('Evaluating on held-out validation set...')
    val_pair_list = []
    retrieved_val_true = 0
    total_val_true = sum(len(v) for v in val_gt.values())

    def process_s1_val_entity(sid):
        cands = get_candidates(sid, top_k=args.top_k)
        cand_ids = [tid for tid, _ in cands]
        retrieved_count = len(val_gt[sid] & set(cand_ids))

        local_pairs = []
        s1_tup = s1_preprocessed[sid][:5]
        for tid, sh in cands:
            if tid in target_preprocessed:
                t_tup = target_preprocessed[tid][:5]
                feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
                local_pairs.append((sid, tid, feats, s1_tup, t_tup))
        return retrieved_count, local_pairs

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        val_results = executor.map(process_s1_val_entity, val_s1_ids, chunksize=200)
        for r_count, l_pairs in val_results:
            retrieved_val_true += r_count
            if l_pairs:
                val_pair_list.extend(l_pairs)

    X_val = np.array([p[2] for p in val_pair_list], dtype=np.float32)
    val_probas = final_model.predict_proba(X_val)

    scores_dict = collections.defaultdict(list)
    for (sid, tid, feats, s1_tup, t_tup), p in zip(val_pair_list, val_probas):
        scores_dict[sid].append((tid, float(p)))

    for sid in val_s1_ids:
        if sid not in scores_dict:
            scores_dict[sid] = []

    # Grid search optimal thresholds with bipartite 1-to-1 consistency
    print(f'Optimizing source-specific thresholds for target metric: {args.target_metric} with 1-to-1 deduplication...')
    best_score = -1.0
    best_s2 = 0.50
    best_s3 = 0.50
    best_metrics = None

    # Pre-sort pairs with proba >= 0.20 once to make grid search 100x faster
    pre_sorted_pairs = []
    for sid, scores in scores_dict.items():
        for tid, p in scores:
            if p >= 0.20:
                pre_sorted_pairs.append((p, sid, tid, tid.startswith('S2-')))
    pre_sorted_pairs.sort(key=lambda x: x[0], reverse=True)

    s2_grid = np.linspace(0.25, 0.85, 61)
    s3_grid = np.linspace(0.25, 0.85, 61)
    for t2 in s2_grid:
        for t3 in s3_grid:
            preds = apply_threshold_and_deduplication(scores_dict, t2, t3, pre_sorted_pairs=pre_sorted_pairs)
            metrics = evaluate_predictions(val_gt, preds)
            score = metrics.get(args.target_metric, metrics['macro_f1'])
            if score > best_score:
                best_score = score
                best_s2 = float(t2)
                best_s3 = float(t3)
                best_metrics = metrics

    opt_preds = apply_threshold_and_deduplication(scores_dict, best_s2, best_s3, pre_sorted_pairs=pre_sorted_pairs)
    final_metrics = evaluate_predictions(val_gt, opt_preds)
    val_cand_recall = retrieved_val_true / total_val_true if total_val_true > 0 else 0.0

    print('\n' + '=' * 60)
    print('FINAL MODEL VALIDATION BENCHMARKS')
    print('=' * 60)
    print(f"Optimal Thresholds: S2 = {best_s2:.2f}, S3 = {best_s3:.2f}")
    print(f"Validation Macro F2   : {final_metrics['macro_f2']:.6f}")
    print(f"Validation Macro F1   : {final_metrics['macro_f1']:.6f}")
    print(f"Validation Macro F0.5 : {final_metrics['macro_f05']:.6f}")
    print(f"Validation Precision  : {final_metrics['global_precision']:.6f}")
    print(f"Validation Recall     : {final_metrics['global_recall']:.6f}")
    print(f"Candidate Recall      : {val_cand_recall:.6f}")
    print(f"False Positives       : {final_metrics['total_fp']}")
    print(f"False Negatives       : {final_metrics['total_fn']}")
    print(f"Singleton Accuracy    : {final_metrics['singleton_accuracy']:.6f}")
    print('=' * 60 + '\n')

    # 8. Save Model and Metadata
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or '.', exist_ok=True)
    final_model.save(args.model_out)

    meta = {
        'optimal_s2_threshold': float(best_s2),
        'optimal_s3_threshold': float(best_s3),
        'validation_f05': float(final_metrics['macro_f05']),
        'validation_precision': float(final_metrics['global_precision']),
        'validation_recall': float(final_metrics['global_recall']),
        'validation_f1': float(final_metrics['macro_f1']),
        'candidate_recall': float(val_cand_recall),
        'false_positives': int(final_metrics['total_fp']),
        'false_negatives': int(final_metrics['total_fn']),
        'singleton_accuracy': float(final_metrics['singleton_accuracy']),
        'model_type': args.model_type,
        'feature_names': FEATURE_NAMES,
        'train_samples': len(train_s1_ids),
        'train_positives': pos_count,
        'train_negatives': neg_count,
        'val_samples': len(val_s1_ids),
        'top_k': args.top_k,
        'seed': args.seed
    }
    with open(args.meta_out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Model saved to {args.model_out} and metadata to {args.meta_out}.')


if __name__ == '__main__':
    main()
