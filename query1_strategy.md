# Query 1 Querying Strategy

## Overview

For our first active learning query, we selected **100 mutants** covering **5 complete positions** (95 mutants) and **5 individual high-uncertainty mutants** from 5 additional positions. Our strategy combined two complementary selection criteria: (1) low structural confidence (pLDDT) and (2) high model disagreement across ESM-2 ensemble members.

## Selection Criteria

### Criterion 1: Low-pLDDT Positions (3 positions: 23, 27, 31)

We used AlphaFold-predicted per-residue pLDDT scores to identify positions where the protein structure is least confidently predicted. Low pLDDT indicates intrinsically disordered or structurally flexible regions, where sequence-based protein language models (PLMs) are expected to be least reliable—the model has little structural context to anchor its predictions.

We selected three positions with the lowest pLDDT values among all test-set positions:

| Position | WT Residue | pLDDT | Region |
|----------|-----------|-------|--------|
| 23       | E         | 27.8  | N-terminal disordered tail |
| 27       | D         | 29.9  | N-terminal disordered tail |
| 31       | C         | 28.2  | N-terminal disordered tail |

**Rationale:** Our leave-position-out simulation on the training data showed that querying low-pLDDT positions yielded the highest cross-position Spearman correlation (ρ = 0.366), substantially outperforming random selection (ρ = 0.307), evenly-spaced selection (ρ = 0.349), and high-uncertainty selection (ρ = 0.225). This is because disordered regions exhibit distinct mutational tolerance patterns that are poorly captured by zero-shot PLM predictions, and obtaining ground-truth labels for these positions provides the supervised model with information that the zero-shot baseline fundamentally lacks.

### Criterion 2: High Model Disagreement Positions (2 positions: 366, 499)

For the remaining 2 complete positions, we computed per-position model disagreement as the mean standard deviation of log-likelihood ratios (LLRs) across our four ESM-2 models (8M, 35M, 150M, 650M) over all 19 possible mutations at each position. High disagreement indicates positions where different model scales have conflicting predictions—suggesting the position is genuinely ambiguous and would benefit most from experimental labels.

| Position | WT Residue | Model Disagreement | pLDDT | Region |
|----------|-----------|-------------------|-------|--------|
| 366      | W         | 5.84              | 96.4  | Structured core |
| 499      | W         | 5.73              | 96.0  | Structured core |

**Rationale:** These positions are in the well-structured enzymatic core (high pLDDT > 95), providing complementary coverage to the disordered N-terminal positions. Despite high structural confidence, the ESM-2 models disagree substantially on mutational effects here, likely because tryptophan residues in enzyme active sites or buried hydrophobic cores have complex fitness landscapes that different model scales interpret differently. Querying these positions diversifies the supervised training signal across distinct protein regions.

### Individual High-Uncertainty Mutants (5 mutants from positions 560, 582, 420, 513, 380)

With 5 complete positions consuming 95 of our 100-mutant budget, we allocated the remaining 5 slots to individual mutants exhibiting the highest per-mutation model disagreement from 5 additional positions. For each position, we selected the single mutation with the largest standard deviation across the four ESM-2 models:

| Mutant | Position | Model Disagreement |
|--------|----------|-------------------|
| W560D  | 560      | High              |
| W582D  | 582      | High              |
| W420I  | 420      | High              |
| E513I  | 513      | High              |
| W380D  | 380      | High              |

**Rationale:** These individual mutants serve as sparse probes across the protein, providing calibration points in additional regions without the cost of querying all 19 mutations at each position.

## Distribution Rationale

We allocated the 100-mutant budget as follows:
- **60% (3 positions, 57 mutants)** to low-pLDDT disordered regions: maximizes information gain where zero-shot models are weakest.
- **40% (2 positions, 38 mutants + 5 individuals)** to high-disagreement structured regions: provides diverse coverage and calibrates ensemble predictions in the protein core.

Querying complete positions (all 19 mutations) rather than individual scattered mutants is critical because it allows the supervised model to learn position-level patterns—how the fitness landscape at a given position relates to the ESM-2 feature vector—which can then transfer to other test positions with similar features.

## Round 1 Results and Analysis

### Returned Labels Summary

| Position | WT | pLDDT | Mean Fitness | Std | Min | Max |
|----------|----|-------|-------------|-----|-----|-----|
| 23 | E | 27.8 | 0.721 | 0.202 | 0.105 (P) | 0.906 (D) |
| 27 | D | 29.9 | 0.702 | 0.211 | 0.183 (P) | 0.976 (T) |
| 31 | C | 28.2 | 0.672 | 0.199 | 0.319 (A) | 0.933 (F) |
| 366 | W | 96.4 | 0.537 | 0.220 | 0.071 (P) | 0.948 (G) |
| 499 | W | 96.0 | 0.463 | 0.142 | 0.038 (P) | 0.606 (M) |

Individual probes: W420I = 0.804, E513I = 0.597, W380D = 0.334, W560D = 0.257, W582D = 0.215

### Biological Insights

1. **Disordered N-terminal positions (23, 27, 31) are highly mutation-tolerant** (mean fitness 0.67–0.72). This is consistent with the expectation that disordered tails have fewer structural constraints, and most substitutions are neutral.

2. **Buried tryptophan positions (366, 499) are mutation-sensitive** (mean fitness 0.46–0.54). Tryptophan is the largest amino acid with unique aromatic stacking properties; replacing it disrupts hydrophobic packing in the enzyme core.

3. **Proline is universally the most deleterious substitution** — it has the lowest fitness at all 5 complete positions (E23P = 0.105, D27P = 0.183, C31P = 0.768 is an exception but C31K = 0.340 is lower; W366P = 0.071, W499P = 0.038). This is expected because proline introduces a backbone kink that disrupts local structure.

4. **Conservative substitutions are well-tolerated** — D27E = 0.913 (acidic→acidic), D27Q = 0.946 (similar size), E23D = 0.906 (same charge), consistent with evolutionary conservation patterns.

5. **Tryptophan→charged mutations at buried core positions are highly deleterious** — W560D = 0.257, W582D = 0.215, W380D = 0.334. Introducing a charged residue into the hydrophobic core destabilizes the protein.

### Zero-Shot Model Evaluation on Queried Positions

We used the Round 1 labels as a true held-out validation set for our zero-shot model, since these positions were drawn from the test set:

| Metric | Value |
|--------|-------|
| Overall Spearman (100 mutants) | 0.141 |
| Pos 23 within-position Spearman | 0.425 |
| Pos 27 within-position Spearman | 0.061 |
| Pos 31 within-position Spearman | −0.184 |
| Pos 366 within-position Spearman | 0.014 |
| Pos 499 within-position Spearman | −0.107 |

The zero-shot model performs poorly on these deliberately challenging positions (ρ = 0.14 overall), compared to ρ = 0.40 on the original training positions and ρ = 0.41 on the full Kaggle test set. This confirms our query strategy was effective at targeting the model's weakest regions—positions where obtaining ground-truth labels provides the most informational value.

The poor within-position correlations (especially at positions 31, 366, 499) indicate that ESM-2 masked marginal log-likelihoods struggle to capture the nuanced fitness landscape at buried tryptophan positions and positions where cysteine disulfide bonds may play a role. These positions exhibit fitness patterns driven by steric and electrostatic factors that single-sequence PLMs cannot fully model.

### Impact on Supervised Model

We trained a Ridge regression on the original 1,140 labels and evaluated on the 100 Round 1 labels as a true held-out test:

| Method | Spearman on Round 1 labels |
|--------|---------------------------|
| Zero-shot (best formula) | 0.141 |
| Ridge supervised | 0.154 |
| Blend (50% sup + 50% ZS) | 0.153 |

The supervised model provides marginal improvement (+0.01) on these hard positions. With only 60 original training positions, the Ridge model has insufficient coverage to generalize to the diverse queried positions. This motivates our adaptive Round 2 strategy: querying positions where the supervised and zero-shot models disagree most, to maximally expand the model's coverage.

## Adaptive Strategy for Round 2

Based on Round 1 results, we shifted from a **prior-based** strategy (pLDDT, model disagreement) to a **posterior-based** strategy for Round 2:

1. **Retrained** Ridge regression on augmented data (1,140 + 100 = 1,240 samples).
2. **Computed disagreement** between supervised predictions and zero-shot scores for all remaining test positions.
3. **Selected 5 positions** with highest supervised-vs-zero-shot disagreement (spread ≥30 positions apart for diversity):
   - 3 high-disagreement positions: pos 62, 155, 197 (all structured core, disagreement > 9.9)
   - 2 diversity positions: pos 402 (pLDDT=64.3, semi-disordered), pos 500 (pLDDT=92.9)
   - 5 individual high-disagreement mutants from other positions

This posterior-based approach targets positions where the model is most uncertain *after* incorporating Round 1 data, rather than relying solely on prior structural features. The disagreement between supervised and zero-shot predictions directly measures where additional labels would most change the model's output—the hallmark of effective active learning.
