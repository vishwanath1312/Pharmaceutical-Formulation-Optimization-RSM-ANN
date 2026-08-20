# Pharmaceutical Formulation Optimization — Research Results Template

## 1. Research Question
State the factor–response modelling and optimization question.

## 2. Experimental Dataset
- Number of runs:
- Factors:
- Responses:
- Factor ranges:
- Missing/outlier observations:

## 3. Preprocessing
Document transformations, scaling and any exclusions.

## 4. RSM / ANOVA
For each response report:
- model equation
- R² / adjusted R² / predicted R² where available
- ANOVA terms and p-values
- lack-of-fit evidence where available
- residual diagnostics

## 5. Predictive Model Benchmark
Use identical validation splits/protocols for all models.
Report MAE, RMSE, R² and other justified metrics per response.

## 6. ANN
Document:
- architecture
- activation
- optimizer
- learning rate
- epochs / stopping rule
- scaling
- validation strategy
- reproducibility seed

## 7. Forward / Backward Pass
Show one representative formulation:
- normalized inputs
- hidden activations
- predicted responses
- target responses
- loss
- output gradients
- hidden gradients

## 8. Multi-Response Optimization
Report:
- optimization objective
- factor bounds
- predicted responses
- desirability
- experimental confirmation status

## 9. Limitations
Discuss sample size, extrapolation risk, model uncertainty and need for experimental confirmation.

## 10. Conclusion
Restrict conclusions to what the computed evidence supports.
