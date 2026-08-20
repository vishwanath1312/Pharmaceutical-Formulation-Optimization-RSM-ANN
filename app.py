import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import differential_evolution
from scipy.stats import zscore

warnings.filterwarnings('ignore')
st.set_page_config(page_title='Formulation Lab | RSM + ANN', page_icon='🧬', layout='wide')

DATA_PATH = os.path.join('data', 'data of the formulation.xlsx')
FEATURES = ['Stearic acid', 'Tween 80', 'Particle size']
RESPONSES = ['Entrapment efficiency', 'Drug content', 'Drug release']

CUSTOM_CSS = '''<style>
.stApp{background:#f6f8fb}
.hero{padding:8px 0 18px}.eyebrow{font-family:monospace;color:#176b87;font-size:.75rem;letter-spacing:.12em;text-transform:uppercase}
.title{font-size:2.15rem;font-weight:750;color:#17212b}.sub{color:#5b6670;font-size:1rem}
.card{background:white;border:1px solid #e4e9ef;border-radius:14px;padding:16px;margin-bottom:12px}
.helpbox{background:#eef7fb;border-left:5px solid #176b87;border-radius:10px;padding:12px 15px;margin:8px 0 16px}
.step{background:white;border:1px solid #e3e8ee;border-radius:12px;padding:12px;height:100%}
.stButton>button,.stDownloadButton>button{border-radius:9px;font-weight:600}
section[data-testid="stSidebar"]{background:#ffffff}
</style>'''
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    required = ['Runs'] + FEATURES + RESPONSES
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    return df[required].dropna().copy()

data = load_data()
X = data[FEATURES]
Y = data[RESPONSES]


def header(title, subtitle=''):
    st.markdown(f'<div class="hero"><div class="eyebrow">Pharmaceutical formulation optimization</div><div class="title">{title}</div><div class="sub">{subtitle}</div></div>', unsafe_allow_html=True)


def metrics_table(actual, pred, model_name):
    rows=[]
    for i,col in enumerate(RESPONSES):
        a=np.asarray(actual)[:,i]; p=np.asarray(pred)[:,i]
        mse=mean_squared_error(a,p)
        rows.append({'Response':col,'Model':model_name,'R²':r2_score(a,p),'MAE':mean_absolute_error(a,p),'RMSE':np.sqrt(mse)})
    return pd.DataFrame(rows)


def rsm_design(df):
    # Full second-order model for 3 continuous factors: A,B,C,A²,B²,C²,AB,AC,BC.
    d=df.copy().rename(columns={'Stearic acid':'A','Tween 80':'B','Particle size':'C'})
    d['A2']=d.A**2; d['B2']=d.B**2; d['C2']=d.C**2; d['AB']=d.A*d.B; d['AC']=d.A*d.C; d['BC']=d.B*d.C
    return d

@st.cache_data
def fit_rsm_all(df):
    d=rsm_design(df)
    models={}; stats={}; anovas={}
    formula='Q("Y") ~ A+B+C+A2+B2+C2+A:B+A:C+B:C'
    for response in RESPONSES:
        dd=d.copy(); dd['Y']=dd[response]
        m=smf.ols(formula, data=dd).fit()
        models[response]=m
        stats[response]={'R²':m.rsquared,'Adjusted R²':m.rsquared_adj,'AIC':m.aic,'BIC':m.bic,'p_model':m.f_pvalue}
        # Type-II ANOVA is appropriate for the fitted quadratic model.
        try:
            from statsmodels.stats.anova import anova_lm
            anovas[response]=anova_lm(m, typ=2).reset_index().rename(columns={'index':'Source'})
        except Exception:
            anovas[response]=pd.DataFrame()
    return models,stats,anovas

rsm_models,rsm_stats,rsm_anovas=fit_rsm_all(data)

@st.cache_resource
def train_ml():
    models={
      'Linear Regression': MultiOutputRegressor(LinearRegression()),
      'Polynomial Regression': Pipeline([('poly',PolynomialFeatures(2,include_bias=False)),('scale',StandardScaler()),('lr',MultiOutputRegressor(LinearRegression()))]),
      'Decision Tree': DecisionTreeRegressor(max_depth=4,random_state=42),
      'Random Forest': RandomForestRegressor(n_estimators=300,max_depth=5,random_state=42),
      'SVR': Pipeline([('scale',StandardScaler()),('svr',MultiOutputRegressor(SVR(C=10,gamma='scale')))]),
      'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(random_state=42,n_estimators=100,max_depth=2,loss='huber')),
    }
    loo=LeaveOneOut(); results=[]; fitted={}
    for name,m in models.items():
        pred=cross_val_predict(m,X,Y,cv=loo)
        results.append(metrics_table(Y,pred,name))
        m.fit(X,Y); fitted[name]=m
    return pd.concat(results,ignore_index=True),fitted

ml_results,ml_models=train_ml()

@st.cache_resource
def train_ann():
    # Article-inspired workflow: scale inputs, MLP, hold out data by leave-one-out CV,
    # then fit the selected architecture on all observations. With 3 inputs/3 responses
    # the natural adaptation of the paper's 4-6-6 MLP is 3-6-3; hidden-size tuning is retained.
    scaler=StandardScaler(); xs=scaler.fit_transform(X)
    candidates=[3,4,5,6,8,10]
    rows=[]; best=None; best_score=np.inf
    loo=LeaveOneOut()
    for h in candidates:
        fold_preds=[]
        for tr,te in loo.split(xs):
            model=MLPRegressor(hidden_layer_sizes=(h,),activation='tanh',solver='lbfgs',alpha=1e-3,max_iter=5000,random_state=42)
            model.fit(xs[tr],Y.iloc[tr])
            fold_preds.append(model.predict(xs[te])[0])
        p=np.asarray(fold_preds); rmse=np.sqrt(mean_squared_error(Y,p))
        rows.append({'Hidden neurons':h,'R²':r2_score(Y,p),'MAE':mean_absolute_error(Y,p),'RMSE':rmse})
        if rmse<best_score: best_score=rmse; best=h
    final=MLPRegressor(hidden_layer_sizes=(best,),activation='tanh',solver='lbfgs',alpha=1e-3,max_iter=5000,random_state=42)
    final.fit(xs,Y)
    loo_pred=[]
    for tr,te in loo.split(xs):
        m=MLPRegressor(hidden_layer_sizes=(best,),activation='tanh',solver='lbfgs',alpha=1e-3,max_iter=5000,random_state=42)
        m.fit(xs[tr],Y.iloc[tr]); loo_pred.append(m.predict(xs[te])[0])
    pred=np.asarray(loo_pred)
    return scaler,final,pd.DataFrame(rows),pred,best
ann_scaler,ann_model,ann_tuning,ann_loo_pred,ann_hidden=train_ann()

st.sidebar.markdown('## 🔬 Formulation Research Lab')
st.sidebar.caption('Research-grade RSM • ML • ANN analysis')
pages=['🏠 Research Overview','📊 Experimental Dataset','📐 DoE / RSM / ANOVA','📈 Response Surfaces','🤖 Predictive Model Benchmark','🧠 ANN Architecture & Training','🔄 ANN Forward / Backward Analysis','🎯 Multi-Response Optimization','🔎 Data Quality & Diagnostics','📑 Research Evidence']
page=st.sidebar.radio('Choose an analysis',pages,index=0)
st.sidebar.markdown('---')
st.sidebar.metric('Experimental runs',len(data))
cA,cB=st.sidebar.columns(2)
cA.metric('Factors',len(FEATURES)); cB.metric('Responses',len(RESPONSES))
st.sidebar.info('**Quick guide**\n\n1. Start with Home\n2. Check Dataset\n3. Explore RSM/ML/ANN\n4. Try an input in Forward & Backward\n5. Review Optimization\n\nResults are model-based and require experimental validation.')

if page=='🏠 Research Overview':
    st.markdown('<div class="hero"><div class="eyebrow">RESEARCH WORKBENCH</div><div class="title">Pharmaceutical Formulation Optimization</div><div class="sub">A reproducible RSM + machine-learning + ANN framework for formulation factor–response modelling.</div></div>',unsafe_allow_html=True)
    st.markdown('### Research question')
    st.write('How do the formulation/process factors influence the measured formulation responses, and can statistical and nonlinear learning models identify a promising multi-response operating region?')
    st.markdown('### Experimental factors and responses')
    c1,c2=st.columns(2)
    with c1:
        st.markdown('**Factors (X)**')
        for f in FEATURES: st.write('•',f)
    with c2:
        st.markdown('**Responses (Y)**')
        for r in RESPONSES: st.write('•',r)
    st.markdown('### Analytical workflow')
    st.code('Experimental data → data audit → scaling/preprocessing → quadratic RSM → ANOVA → residual diagnostics → ML benchmark → ANN training/tuning → forward/backward analysis → desirability optimization → validation')
    st.markdown('### Research safeguards')
    st.warning('The supplied dataset contains 10 experimental runs. Model complexity, validation strategy and optimization claims should therefore be interpreted cautiously. A model-derived optimum is a hypothesis for experimental confirmation, not a validated pharmaceutical formulation.')
    st.markdown('### Reproducibility')
    st.write('All analyses should be run from the supplied dataset with fixed preprocessing, documented model settings, validation metrics and saved figures/tables. Avoid reporting training performance alone.')

elif page=='📊 Experimental Dataset':
    header('Dataset','The application uses the newly supplied formulation workbook as the single source of experimental data.')
    st.dataframe(data,width='stretch',hide_index=True)
    st.markdown('### Descriptive statistics')
    st.dataframe(data[FEATURES+RESPONSES].describe().T,width='stretch')
    csv=data.to_csv(index=False).encode()
    st.download_button('Download processed dataset CSV',csv,'processed_formulation_dataset.csv','text/csv')

elif page=='📐 DoE / RSM / ANOVA':
    header('Quadratic RSM and ANOVA','Second-order response-surface model with main effects, two-factor interactions and quadratic terms.')
    response=st.selectbox('Response',RESPONSES)
    m=rsm_models[response]
    s=rsm_stats[response]
    a=rsm_anovas[response]
    c1,c2,c3=st.columns(3); c1.metric('R²',f"{s['R²']:.4f}"); c2.metric('Adjusted R²',f"{s['Adjusted R²']:.4f}"); c3.metric('Model p-value',f"{s['p_model']:.4g}")
    st.markdown('#### ANOVA table')
    st.dataframe(a,width='stretch',hide_index=True)
    st.markdown('#### Model summary')
    st.text(m.summary())
    st.caption('Because the supplied dataset has only 10 runs and the full 3-factor quadratic model has 9 fitted coefficients, residual degrees of freedom are extremely limited. Treat p-values and individual coefficients as exploratory and add more designed experimental runs before making confirmatory claims.')

elif page=='📈 Response Surfaces':
    header('Response Surfaces','3D/contour visualization for the fitted quadratic RSM. One factor is held at its observed mean.')
    response=st.selectbox('Response',RESPONSES)
    pairs=[('Stearic acid','Tween 80','Particle size'),('Stearic acid','Particle size','Tween 80'),('Tween 80','Particle size','Stearic acid')]
    pair=st.selectbox('Factor pair', [f'{a} × {b} (hold {c} at mean)' for a,b,c in pairs])
    a,b,c=pairs[[f'{x} × {y} (hold {z} at mean)' for x,y,z in pairs].index(pair)]
    g1=np.linspace(X[a].min(),X[a].max(),35); g2=np.linspace(X[b].min(),X[b].max(),35); G1,G2=np.meshgrid(g1,g2)
    grid=pd.DataFrame({a:G1.ravel(),b:G2.ravel(),c:X[c].mean()})[FEATURES]
    pred=rsm_models[response].predict(rsm_design(pd.concat([grid],axis=1)).assign(Y=0)) if False else None
    # Predict using statsmodels with original factor names.
    pred=rsm_models[response].predict(rsm_design(grid.assign(**{response:0})).assign(Y=0))
    Z=np.asarray(pred).reshape(G1.shape)
    fig=go.Figure(go.Surface(x=g1,y=g2,z=Z,colorscale='Viridis',colorbar=dict(title=response)))
    fig.add_trace(go.Scatter3d(x=X[a],y=X[b],z=Y[response],mode='markers',marker=dict(size=6,color='orange'),name='Experimental'))
    fig.update_layout(scene=dict(xaxis_title=a,yaxis_title=b,zaxis_title=response),height=620)
    st.plotly_chart(fig,width='stretch')
    fig2=go.Figure(go.Contour(x=g1,y=g2,z=Z,colorscale='Viridis',contours=dict(showlabels=True)))
    fig2.add_trace(go.Scatter(x=X[a],y=X[b],mode='markers',marker=dict(size=9,color='orange',symbol='x'),name='Experimental'))
    fig2.update_layout(xaxis_title=a,yaxis_title=b,height=500)
    st.plotly_chart(fig2,width='stretch')

elif page=='🤖 Predictive Model Benchmark':
    header('Machine Learning Comparison','Leave-one-out cross-validation is used because the supplied dataset contains only 10 observations.')
    summary=ml_results.groupby('Model')[['R²','MAE','RMSE']].mean().sort_values(['RMSE','MAE'])
    st.dataframe(summary.style.format({'R²':'{:.4f}','MAE':'{:.5f}','RMSE':'{:.5f}'}),width='stretch')
    model=st.selectbox('Detailed model',summary.index.tolist())
    st.dataframe(ml_results[ml_results.Model==model],width='stretch',hide_index=True)
    st.caption('For a small experimental dataset, cross-validation estimates can be unstable. Use these results for model comparison, not as a substitute for independent experimental validation.')

elif page=='🧠 ANN Architecture & Training':
    header('Artificial Neural Network','Article-inspired ANN workflow: scaled inputs, hidden-neuron trial-and-error, leave-one-out validation and final prediction model.')
    st.write(f'The reference paper reported an optimum **4-6-6 MLP** for four inputs and six outputs. Here, the supplied dataset has **3 inputs and 3 outputs**, so the architecture is adapted to **3-{ann_hidden}-3**, with the hidden layer selected by minimum LOOCV RMSE.')
    st.markdown('### Hidden-layer tuning')
    st.dataframe(ann_tuning,width='stretch',hide_index=True)
    st.success(f'Selected ANN architecture: 3-{ann_hidden}-3')
    ann_metrics=metrics_table(Y,ann_loo_pred,f'ANN 3-{ann_hidden}-3')
    st.dataframe(ann_metrics,width='stretch',hide_index=True)
    comp=pd.concat([ml_results,ann_metrics],ignore_index=True)
    st.markdown('### RSM vs ANN/ML validation')
    st.dataframe(comp.groupby('Model')[['R²','MAE','RMSE']].mean().sort_values('RMSE'),width='stretch')
    st.caption('The supplied article reports that ANN can outperform second-order RSM when relationships are nonlinear. That is a methodological motivation, not a guarantee for this 10-run dataset.')

elif page=='🔄 ANN Forward / Backward Analysis':
    header('ANN Interactive Forward & Backward Pass','Enter a formulation, perform forward propagation through the trained ANN, calculate loss, and inspect the backward-pass gradients.')
    st.markdown('<div class="helpbox"><b>How to use this page</b><br>① Enter three formulation values within the experimental range → ② click through the prediction → ③ choose an experimental run as the target → ④ inspect the error and backward gradients. This is an educational view of how the ANN learns.</div>', unsafe_allow_html=True)
    st.markdown('### Step 1 — Enter formulation values')
    preset = st.selectbox('Input preset', ['Use mean values','Use an experimental run'])
    selected_run = None
    if preset == 'Use an experimental run':
        selected_run = st.selectbox('Choose experimental run', data['Runs'].tolist(), key='input_run')
    cols=st.columns(3)
    input_values=[]
    for i,f in enumerate(FEATURES):
        lo=float(X[f].min()); hi=float(X[f].max())
        default=float(data.loc[data['Runs']==selected_run,f].iloc[0]) if selected_run is not None else float(X[f].mean())
        input_values.append(cols[i].number_input(f'**{f}**', min_value=lo, max_value=hi, value=default, step=(hi-lo)/100 if hi>lo else 1.0, format='%.4f', help=f'Allowed range: {lo:.4f} to {hi:.4f}'))
    x_raw=np.asarray(input_values,dtype=float).reshape(1,-1)
    st.caption('Tip: staying inside the observed experimental range avoids extrapolation.')

    st.markdown('### Step 2 — Forward propagation')
    st.caption('The ANN converts your three inputs into three predicted responses.')
    x_scaled=ann_scaler.transform(x_raw)
    W1=ann_model.coefs_[0]; b1=ann_model.intercepts_[0]
    W2=ann_model.coefs_[1]; b2=ann_model.intercepts_[1]
    z1=x_scaled @ W1 + b1
    h=np.tanh(z1)
    z2=h @ W2 + b2
    yhat=z2
    st.write('**Scaled input X:**')
    st.dataframe(pd.DataFrame(x_scaled,columns=FEATURES),hide_index=True,width='stretch')
    st.write('**Hidden-layer pre-activation Z₁:**')
    st.dataframe(pd.DataFrame(z1,columns=[f'H{i+1}' for i in range(W1.shape[1])]),hide_index=True,width='stretch')
    st.write('**Hidden-layer activation H = tanh(Z₁):**')
    st.dataframe(pd.DataFrame(h,columns=[f'H{i+1}' for i in range(W1.shape[1])]),hide_index=True,width='stretch')
    st.write('**Predicted responses Ŷ:**')
    pred_cols=st.columns(3)
    for i,r in enumerate(RESPONSES):
        pred_cols[i].metric(r, f'{yhat[0,i]:.3f}')
    st.markdown('<div class="helpbox">These are <b>model predictions</b>, not laboratory measurements.</div>', unsafe_allow_html=True)

    st.markdown('### Step 3 — Compare with an experimental target')
    st.caption('Choose a measured run to calculate the prediction error. The saved ANN is not changed.')
    run_id=st.selectbox('Target experimental run',data['Runs'].tolist(),index=0,key='target_run')
    target=data.loc[data['Runs']==run_id,RESPONSES].to_numpy(dtype=float)
    error=yhat-target
    loss=float(np.mean(error**2))
    dL_dyhat=(2.0/len(RESPONSES))*error
    dZ2=dL_dyhat
    dW2=h.T @ dZ2
    db2=dZ2[0]
    dH=dZ2 @ W2.T
    dZ1=dH*(1.0-h**2)
    dW1=x_scaled.T @ dZ1
    db1=dZ1[0]
    st.metric('MSE Loss',f'{loss:.8f}')
    st.write('**Target Y and output error (Ŷ − Y):**')
    st.dataframe(pd.DataFrame({'Response':RESPONSES,'Target Y':target[0],'Predicted Ŷ':yhat[0],'Error':error[0],'dL/dŶ':dL_dyhat[0]}),hide_index=True,width='stretch')
    st.markdown('### Step 4 — Backward propagation')
    st.success('The error is now propagated backward through the ANN so we can see which weights and hidden neurons receive the learning signal.')
    advanced = st.toggle('Show detailed gradient matrices and equations', value=False)
    if advanced:
        c1,c2=st.columns(2)
        with c1:
            st.write('**Output-layer gradients**')
            st.dataframe(pd.DataFrame(dW2,index=[f'H{i+1}' for i in range(W2.shape[0])],columns=RESPONSES),width='stretch')
            st.write('Output bias gradient')
            st.dataframe(pd.DataFrame([db2],columns=RESPONSES),hide_index=True,width='stretch')
        with c2:
            st.write('**Hidden-layer gradients**')
            st.dataframe(pd.DataFrame(dW1,index=FEATURES,columns=[f'H{i+1}' for i in range(W1.shape[1])]),width='stretch')
            st.write('Hidden bias gradient')
            st.dataframe(pd.DataFrame([db1],columns=[f'H{i+1}' for i in range(W1.shape[1])]),hide_index=True,width='stretch')

        st.markdown('### Backward-pass equations')
    st.latex(r'Z_1=XW_1+b_1,\quad H=\tanh(Z_1),\quad \hat{Y}=HW_2+b_2')
    st.latex(r'L=\frac{1}{3}\sum_j(\hat{Y}_j-Y_j)^2')
    st.latex(r'\delta_2=\frac{2}{3}(\hat{Y}-Y),\quad \nabla W_2=H^T\delta_2,\quad \nabla b_2=\delta_2')
    st.latex(r'\delta_1=(\delta_2W_2^T)\odot(1-H^2),\quad \nabla W_1=X^T\delta_1,\quad \nabla b_1=\delta_1')
    st.success('The backward pass shows how the prediction error is propagated from Entrapment efficiency, Drug content and Drug release back to every hidden neuron and input-to-hidden weight.')

elif page=='🎯 Multi-Response Optimization':
    header('Multi-response Optimization','Model-based search that maximizes the three recorded response values simultaneously.')
    st.markdown('<div class="helpbox"><b>Goal:</b> find input values that give a strong combined prediction for all three responses. The result is a model-based recommendation that must be tested experimentally.</div>', unsafe_allow_html=True)
    st.write('A geometric desirability is used for the three responses: each response is normalized between its observed minimum and maximum, then combined by the geometric mean. Particle size is an input factor in this dataset, not an output.')
    mins=Y.min(); maxs=Y.max()
    def desirability(v):
        ds=[]
        for i,col in enumerate(RESPONSES):
            den=maxs[col]-mins[col]
            ds.append(0.0 if den==0 else np.clip((v[i]-mins[col])/den,0,1))
        return float(np.prod(ds)**(1/len(ds)))
    model_name=st.selectbox('Optimization model',['RSM','Random Forest','ANN'],index=0)
    if model_name=='RSM':
        def pred(v):
            row=pd.DataFrame([v],columns=FEATURES); rd=rsm_design(row.assign(**{RESPONSES[0]:0})); return np.array([rsm_models[c].predict(rd.assign(Y=0))[0] for c in RESPONSES])
    elif model_name=='Random Forest':
        rf=ml_models['Random Forest']; pred=lambda v: rf.predict(pd.DataFrame([v],columns=FEATURES))[0]
    else:
        pred=lambda v: ann_model.predict(ann_scaler.transform(pd.DataFrame([v],columns=FEATURES)))[0]
    bounds=[(X[c].min(),X[c].max()) for c in FEATURES]
    res=differential_evolution(lambda v:-desirability(pred(v)),bounds,seed=42,popsize=12,maxiter=120,polish=True)
    best=res.x; pv=pred(best); dscore=desirability(pv)
    c1,c2,c3,c4=st.columns(4); c1.metric('Stearic acid',f'{best[0]:.3f}'); c2.metric('Tween 80',f'{best[1]:.3f}'); c3.metric('Particle size',f'{best[2]:.3f}'); c4.metric('Desirability',f'{dscore:.4f}')
    opt=pd.DataFrame({'Variable':FEATURES+RESPONSES+['Overall desirability'],'Value':list(best)+list(pv)+[dscore]})
    st.dataframe(opt,width='stretch',hide_index=True)
    # Nearest experimental run for validation context.
    dist=((X[FEATURES]-best)/X[FEATURES].std()).pow(2).sum(axis=1)
    idx=dist.idxmin()
    st.markdown('### Nearest experimental formulation')
    st.dataframe(data.loc[[idx]],width='stretch',hide_index=True)
    st.caption('Optimization is a model-based recommendation. It must be experimentally prepared and validated before any pharmaceutical conclusion is drawn.')

elif page=='🔎 Data Quality & Diagnostics':
    header('Outlier Analysis','Simple z-score screening for data-quality review; flagged observations are not automatically deleted.')
    num=data[FEATURES+RESPONSES]
    z=np.abs(zscore(num,ddof=0)); flags=(z>2.5).any(axis=1)
    st.metric('Flagged runs',int(flags.sum()))
    st.dataframe(data.assign(Outlier=flags),width='stretch',hide_index=True)
    fig=make_subplots(rows=2,cols=3,subplot_titles=FEATURES+RESPONSES)
    for i,col in enumerate(FEATURES+RESPONSES):
        r=i//3+1;c=i%3+1
        fig.add_trace(go.Box(y=data[col],name=col,boxmean=True),row=r,col=c)
    fig.update_layout(height=650,showlegend=False)
    st.plotly_chart(fig,width='stretch')


elif page=='📑 Research Evidence':
    st.markdown('## 📑 Research Evidence & Reporting')
    st.markdown('<div class="helpbox"><b>Purpose:</b> convert computational results into defensible research evidence. Report experimental design, preprocessing, model specification, validation, uncertainty/diagnostics and optimization separately.</div>',unsafe_allow_html=True)
    tabs=st.tabs(['Study design','Model reporting','Validation','Optimization reporting','Article alignment'])
    with tabs[0]:
        st.markdown('**Experimental design record**')
        st.write(f'Number of runs: **{len(data)}**')
        st.write('Factors: ' + ', '.join(FEATURES))
        st.write('Responses: ' + ', '.join(RESPONSES))
        st.info('Do not infer or label a formal DoE design (e.g., Box–Behnken) unless the supplied run structure and factor levels support that claim.')
    with tabs[1]:
        st.markdown('**Minimum model-reporting checklist**')
        for x in ['Preprocessing/scaling method','Model architecture and activation function','Hyperparameters and stopping criteria','Training/validation strategy','Random seed where applicable','Performance metrics for every response','Predicted-vs-experimental plots','Residual/error diagnostics']:
            st.checkbox(x, value=False, key='check_'+re.sub('[^a-zA-Z0-9]','_',x))
    with tabs[2]:
        st.markdown('**Validation principles**')
        st.write('For a very small experimental dataset, use validation methods appropriate to the sample size (for example leave-one-out cross-validation where implemented), and report fold-wise or aggregated metrics. Compare models on the same splits.')
        st.warning('Do not claim generalization, superiority of ANN over RSM, or statistical significance unless supported by the computed validation evidence.')
    with tabs[3]:
        st.markdown('**Optimization evidence**')
        st.write('Report the objective/desirability definition, factor bounds, predicted responses at the optimum, desirability score, and—most importantly—experimental confirmation of the proposed formulation.')
    with tabs[4]:
        st.markdown('**Supplied article alignment**')
        st.write('The supplied article motivates the combined use of design-of-experiments/RSM and ANN for formulation/process modelling. This implementation adapts that workflow to the supplied three-factor/three-response dataset; it does not reproduce the article experimental design when the supplied dataset does not support it.')
 st.markdown('---')
st.caption('Pharmaceutical Formulation Optimization Dashboard • RSM + ML + ANN • For research and educational use; model recommendations require experimental validation.')
