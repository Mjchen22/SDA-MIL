import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_path', required=True,
                    help='Camelyon16 dataset CSV with case_id, slide_id, label columns')
    ap.add_argument('--task', default='task_4_camelyon16',
                    help='task name; used to build splits/<task>_100 output dir')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--k', type=int, default=5, help='number of folds')
    ap.add_argument('--label_frac', type=float, default=1.0)
    ap.add_argument('--test_prefix', default='test_',
                    help='slide_id prefix that identifies the held-out official test set')
    ap.add_argument('--label_normal', default='normal_tissue',
                    help='string label for normal slides in the CSV')
    ap.add_argument('--label_tumor', default='tumor_tissue',
                    help='string label for tumor slides in the CSV')
    ap.add_argument('--out_root', default='splits')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv_path)
    required_cols = {'case_id', 'slide_id', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must contain columns {required_cols}, got {df.columns.tolist()}')

    # Sanity check label values
    label_set = set(df['label'].unique())
    expected = {args.label_normal, args.label_tumor}
    if label_set != expected:
        # Try to be helpful before failing
        if {'Normal', 'Tumor'} == label_set:
            print(f"WARN: CSV has labels {label_set} but main.py expects "
                  f"{{'normal_tissue', 'tumor_tissue'}}. "
                  f"Re-run with --label_normal Normal --label_tumor Tumor "
                  f"to use these labels, OR rename them in the CSV first.")
        raise ValueError(f"CSV has labels {label_set}, expected {expected}. "
                         f"Use --label_normal/--label_tumor to override if needed.")

    # Partition: official test set is fixed by slide_id prefix
    is_test = df['slide_id'].str.startswith(args.test_prefix)
    test_df = df[is_test].reset_index(drop=True)
    train_pool = df[~is_test].reset_index(drop=True)

    n_test = len(test_df)
    n_pool = len(train_pool)
    print(f"\n=== Camelyon16 splitter ===")
    print(f"Total slides:                    {len(df)}")
    print(f"Official test (fixed all folds): {n_test}")
    print(f"  test  normal: {(test_df['label'] == args.label_normal).sum()}")
    print(f"  test  tumor:  {(test_df['label'] == args.label_tumor).sum()}")
    print(f"Train pool (to k-fold split):    {n_pool}")
    print(f"  pool  normal: {(train_pool['label'] == args.label_normal).sum()}")
    print(f"  pool  tumor:  {(train_pool['label'] == args.label_tumor).sum()}")

    # Class-stratified k-fold partition of the train pool.
    # Each train-pool slide belongs to exactly one fold's val set.
    pool_normal = train_pool[train_pool['label'] == args.label_normal].sample(
        frac=1.0, random_state=args.seed).reset_index(drop=True)
    pool_tumor = train_pool[train_pool['label'] == args.label_tumor].sample(
        frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    # Split each class shuffled list into k near-equal chunks.
    def split_into_k(s, k):
        n = len(s)
        sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        out = []
        idx = 0
        for sz in sizes:
            out.append(s.iloc[idx:idx + sz].reset_index(drop=True))
            idx += sz
        return out

    val_chunks_normal = split_into_k(pool_normal, args.k)
    val_chunks_tumor = split_into_k(pool_tumor, args.k)

    # Compose folds
    out_dir = os.path.join(args.out_root, f'{args.task}_{int(args.label_frac * 100)}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nWriting splits to: {out_dir}")

    for fold in range(args.k):
        val_df = pd.concat([val_chunks_normal[fold], val_chunks_tumor[fold]],
                           ignore_index=True)
        train_mask = ~train_pool['slide_id'].isin(val_df['slide_id'])
        train_df = train_pool[train_mask].reset_index(drop=True)

        # If label_frac < 1, subsample train (per CLAM convention)
        if args.label_frac < 1.0:
            n_keep = int(round(len(train_df) * args.label_frac))
            train_df = train_df.sample(n=n_keep, random_state=args.seed + fold).reset_index(drop=True)

        # ---- Long-format split file: column-aligned train/val/test slide_ids ----
        max_len = max(len(train_df), len(val_df), len(test_df))
        long_df = pd.DataFrame({
            'train': pd.Series(train_df['slide_id'].tolist() + [''] * (max_len - len(train_df))),
            'val':   pd.Series(val_df['slide_id'].tolist()   + [''] * (max_len - len(val_df))),
            'test':  pd.Series(test_df['slide_id'].tolist()  + [''] * (max_len - len(test_df))),
        })
        long_df.to_csv(os.path.join(out_dir, f'splits_{fold}.csv'), index=True)

        # ---- Boolean-style split file ----
        all_slides = pd.concat([train_df, val_df, test_df], ignore_index=True)['slide_id']
        bool_df = pd.DataFrame({'slide_id': all_slides})
        bool_df['train'] = bool_df['slide_id'].isin(train_df['slide_id'])
        bool_df['val']   = bool_df['slide_id'].isin(val_df['slide_id'])
        bool_df['test']  = bool_df['slide_id'].isin(test_df['slide_id'])
        bool_df.to_csv(os.path.join(out_dir, f'splits_{fold}_bool.csv'), index=False)

        # ---- Descriptor: per-class counts ----
        desc = pd.DataFrame({
            'train': [
                (train_df['label'] == args.label_normal).sum(),
                (train_df['label'] == args.label_tumor).sum(),
            ],
            'val': [
                (val_df['label'] == args.label_normal).sum(),
                (val_df['label'] == args.label_tumor).sum(),
            ],
            'test': [
                (test_df['label'] == args.label_normal).sum(),
                (test_df['label'] == args.label_tumor).sum(),
            ],
        }, index=[args.label_normal, args.label_tumor])
        desc.to_csv(os.path.join(out_dir, f'splits_{fold}_descriptor.csv'))

        print(f"  fold {fold}: train={len(train_df):3d} "
              f"(N={ (train_df['label']==args.label_normal).sum() }, "
              f"T={ (train_df['label']==args.label_tumor).sum() })  "
              f"val={len(val_df):3d} "
              f"(N={ (val_df['label']==args.label_normal).sum() }, "
              f"T={ (val_df['label']==args.label_tumor).sum() })  "
              f"test={len(test_df):3d}")

    # Cross-fold sanity check: each train-pool slide should appear in val exactly once.
    val_appearances = {}
    for fold in range(args.k):
        val_set = set(pd.concat([val_chunks_normal[fold], val_chunks_tumor[fold]],
                                ignore_index=True)['slide_id'])
        for sid in val_set:
            val_appearances[sid] = val_appearances.get(sid, 0) + 1
    counts = pd.Series(val_appearances).value_counts()
    print(f"\nCross-fold val coverage: each train-pool slide appears in val {dict(counts)} folds "
          f"(should all be 1).")

    print("\nDone.")


if __name__ == '__main__':
    main()