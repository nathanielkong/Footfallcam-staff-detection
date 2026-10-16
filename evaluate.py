"""Evaluate presence on explicitly labelled frames only; never infer missing labels."""
import argparse
import json
from pathlib import Path
import pandas as pd


def evaluate(results, labels=None):
    if labels is None:
        return {'status':'not_evaluated', 'reason':'No reviewed frame labels supplied. No accuracy claim is made.'}
    truth = pd.read_csv(labels)
    required = {'frame','staff_present'}
    if not required.issubset(truth.columns):
        raise ValueError('Labels require frame,staff_present columns')
    if truth['frame'].duplicated().any() or not truth['staff_present'].isin([0,1]).all():
        raise ValueError('Labels must have unique frame indices and binary staff_present values')
    if not (truth['frame'] >= 0).all() or not (truth['frame'] % 1 == 0).all():
        raise ValueError('Label frame indices must be nonnegative integers')
    merged = results.merge(truth[['frame','staff_present']], on='frame', suffixes=('_pred','_true'), how='inner', validate='one_to_one')
    if len(merged) != len(truth) or len(merged) == 0:
        raise ValueError('Every labelled frame must exist in the results')
    pred, actual = merged.staff_present_pred.astype(bool), merged.staff_present_true.astype(bool)
    tp, fp = int((pred & actual).sum()), int((pred & ~actual).sum())
    fn, tn = int((~pred & actual).sum()), int((~pred & ~actual).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {'status':'evaluated', 'labels':str(Path(labels).resolve()), 'labelled_frames':len(merged),
            'total_frames':len(results), 'true_positive':tp,'false_positive':fp,'false_negative':fn,'true_negative':tn,
            'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.0,
            'note':'Frame-presence metrics only. Label provenance and holdout policy must be checked separately. Not a tag-localization metric.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('metrics.json'))
    args=parser.parse_args()
    metrics=evaluate(pd.read_csv(args.results),args.labels)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
