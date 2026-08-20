# Pharmaceutical Formulation Optimization — RSM + ML + ANN

This version uses the newly supplied formulation dataset and follows the methodological logic of the supplied Pharmaceutics RSM + ANN article.

## Dataset
**Inputs:** Stearic acid, Tween 80, Particle size  
**Responses:** Entrapment efficiency, Drug content, Drug release  
**Experimental runs:** 10

## Methodology
1. Dataset validation and descriptive statistics
2. Quadratic Response Surface Methodology (RSM)
3. ANOVA for main, interaction and quadratic effects
4. 3D response-surface and contour plots
5. ML comparison using leave-one-out cross-validation
6. ANN with scaled inputs and hidden-neuron tuning
7. Interactive ANN forward pass and backward-pass gradient analysis
8. Multi-response desirability optimization
9. Experimental-run proximity/validation check
10. Outlier screening

## Article alignment
The supplied article uses RSM/DoE to establish formulation/process relationships and an ANN to model nonlinear relationships and support optimization. It reports a 4-6-6 MLP because its study has four inputs and six outputs. This implementation adapts that architecture to the supplied dataset: 3 inputs → tuned hidden layer → 3 outputs. The application deliberately does **not** call the 10-run supplied dataset a Box–Behnken design.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```


## Interactive ANN forward and backward pass
The dashboard includes an **ANN Forward & Backward Pass** page. The user enters Stearic acid, Tween 80 and Particle size within the observed training ranges. The application performs forward propagation through the trained ANN and displays scaled inputs, hidden-layer pre-activation, tanh hidden activations and the three predicted responses.

For backward propagation, an experimental run is selected as the target. The application calculates MSE loss and propagates the error backward using the standard chain rule for the single-hidden-layer tanh MLP. It displays output-layer and hidden-layer weight/bias gradients. The saved ANN is not changed; this is an analysis/teaching view of the trained network.


## User-Friendly Workflow

1. Open the dashboard with `streamlit run app.py`.
2. Start at **Home** for the recommended workflow.
3. Use **Dataset** to inspect/download the supplied formulation data.
4. Use **RSM & ANOVA** and **Response Surfaces** to understand factor-response relationships.
5. Use **ML Comparison** and **ANN Model** to compare predictive approaches.
6. Use **ANN Forward & Backward**:
   - enter Stearic acid, Tween 80 and Particle size;
   - view the three predictions;
   - choose an experimental run as the target;
   - inspect MSE loss;
   - enable the advanced toggle to view gradients and equations.
7. Use **Optimization** for model-based formulation recommendations.
8. Use **Data Quality** to review possible outliers.

The application is intended for research/education. Optimization outputs must be experimentally validated before pharmaceutical conclusions are made.


# Research-Oriented Workflow

This version is intended as a research/academic workbench rather than a simple prediction demo.

## Recommended reporting sequence

1. **Experimental Dataset** — document factor levels, responses, missing values and data quality.
2. **DoE / RSM / ANOVA** — fit the quadratic response-surface model and inspect statistical evidence.
3. **Response Surfaces** — interpret factor interactions and curvature.
4. **Predictive Model Benchmark** — compare candidate ML models using the same validation protocol.
5. **ANN Architecture & Training** — document architecture, activation, scaling, training settings and performance.
6. **ANN Forward / Backward Analysis** — demonstrate prediction and gradient propagation for an individual formulation.
7. **Multi-Response Optimization** — report the objective, bounds, predicted responses and desirability.
8. **Research Evidence** — use the checklist before writing results/conclusions.

## Research reporting principles

- Separate training performance from validation performance.
- Do not infer a formal DoE design unless supported by the supplied experiments.
- Avoid overclaiming from a 10-run dataset.
- Treat optimization as a model-derived candidate until experimentally validated.
- Preserve the exact dataset, preprocessing, model settings and validation protocol for reproducibility.
