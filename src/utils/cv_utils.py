"""
Cross-validation utilities for MATWI set-based LOSO-CV.

The 13 mapped sets (1-13) are used. Sets 14-17 are excluded (unvalidated).
Val rotation: for fold i, test=sets[i], val=sets[(i+1) % 13], train=remaining 11.
"""

ALL_SETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def make_loso_folds(sets: list[int] = ALL_SETS) -> list[dict]:
    """
    Generate leave-one-set-out folds with rotating val set.

    Returns a list of dicts, one per fold:
        {"test": [set_num], "val": [set_num], "train": [set_num, ...]}
    """
    folds = []
    for i, test_set in enumerate(sets):
        val_set    = sets[(i + 1) % len(sets)]
        train_sets = [s for s in sets if s != test_set and s != val_set]
        folds.append({
            "fold":  i,
            "test":  [test_set],
            "val":   [val_set],
            "train": train_sets,
        })
    return folds


if __name__ == "__main__":
    for f in make_loso_folds():
        print(f"Fold {f['fold']:2d}  test={f['test']}  val={f['val']}  train={f['train']}")
