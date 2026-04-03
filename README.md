# MLB S26 Hackathon (Group 2)

## Requirements

```bash
pip install fair-esm torch numpy pandas scikit-learn scipy
```

## Quick Run

```bash
bash reproduce_best.sh
```

## From Scratch

```bash
bash reproduce_from_scratch.sh
```

## Submission Packaging

```bash
cp predictions_q1cal_w30.csv predictions.csv
zip -j submission.zip APIKey.txt GroupName.txt predictions.csv
```
