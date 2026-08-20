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
7. Multi-response desirability optimization
8. Experimental-run proximity/validation check
9. Outlier screening

## Article alignment
The supplied article uses RSM/DoE to establish formulation/process relationships and an ANN to model nonlinear relationships and support optimization. It reports a 4-6-6 MLP because its study has four inputs and six outputs. This implementation adapts that architecture to the supplied dataset: 3 inputs → tuned hidden layer → 3 outputs. The application deliberately does **not** call the 10-run supplied dataset a Box–Behnken design.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
