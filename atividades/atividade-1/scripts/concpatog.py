import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

plt.style.use('seaborn-v0_8-whitegrid')

### Carregamento
df = pd.read_csv('atividades/atividade-1/datasets/ConcPatog.txt', sep='\t')
df.columns = df.columns.str.strip()
n = len(df)
print(df.shape)
print(df[['CPatog1', 'CPatog2']].describe())
print("Valores ausentes:\n", df[['CPatog1', 'CPatog2']].isna().sum())
print("Zeros em CPatog1:", (df.CPatog1 == 0).sum(), "| Zeros em CPatog2:", (df.CPatog2 == 0).sum())

# Dispersão
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(df.CPatog1, df.CPatog2, s=50, alpha=0.7, color='steelblue', edgecolors='white')
ax.set_xlabel('CPatog1 (concentração do patógeno 1)'); ax.set_ylabel('CPatog2 (concentração do patógeno 2)')
ax.set_title('Dispersão: CPatog2 vs. CPatog1')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/concpatog/cp_fig1_dispersao.png', dpi=150); plt.close()

# Ajuste
m = smf.ols('CPatog2 ~ CPatog1', data=df).fit()
print(m.summary())
print("R2 =", m.rsquared, " R2adj =", m.rsquared_adj)

# Reta + IC + IP
xr = pd.DataFrame({'CPatog1': np.linspace(df.CPatog1.min(), df.CPatog1.max(), 200)})
sf = m.get_prediction(xr).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(df.CPatog1, df.CPatog2, s=50, alpha=0.7, color='steelblue', edgecolors='white', label='Dados', zorder=3)
ax.plot(xr.CPatog1, sf['mean'], color='crimson', lw=2.5, label='Reta de regressão')
ax.fill_between(xr.CPatog1, sf.mean_ci_lower, sf.mean_ci_upper, color='crimson', alpha=0.25, label='IC 95% (média)')
ax.fill_between(xr.CPatog1, sf.obs_ci_lower, sf.obs_ci_upper, color='gray', alpha=0.15, label='IP 95% (nova obs.)')
ax.set_xlabel('CPatog1'); ax.set_ylabel('CPatog2'); ax.legend()
ax.set_title(f'Reta de regressão com IC e IP (R² = {m.rsquared:.3f})')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/concpatog/cp_fig2_reta_ic_ip.png', dpi=150); plt.close()

### Resíduos
res = m.resid; fit = m.fittedvalues

# Histograma + densidade normal teórica
fig, ax = plt.subplots(figsize=(8,5))
bins = max(6, int(n/10))
ax.hist(res, bins=bins, density=True, color='steelblue', edgecolor='white', alpha=0.8, label='Resíduos')
xs = np.linspace(res.min(), res.max(), 400)
ax.plot(xs, stats.norm.pdf(xs, res.mean(), res.std(ddof=1)), color='crimson', lw=2.5, label='Normal teórica')
ax.set_xlabel('Resíduos'); ax.set_ylabel('Densidade'); ax.legend()
ax.set_title('Histograma dos resíduos')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/concpatog/cp_fig3_hist_res.png', dpi=150); plt.close()

# Resíduos vs. preditos
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(fit, res, s=50, alpha=0.7, color='steelblue', edgecolors='white')
ax.axhline(0, color='crimson', ls='--')
ax.set_xlabel('Valores preditos'); ax.set_ylabel('Resíduos'); ax.set_title('Resíduos vs. valores preditos')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/concpatog/cp_fig4_res_vs_pred.png', dpi=150); plt.close()

# QQ-plot
fig, ax = plt.subplots(figsize=(6,6))
sm.qqplot(res, line='s', ax=ax, markerfacecolor='steelblue', markeredgecolor='white', markersize=7)
ax.set_title('QQ-plot dos resíduos (normal)')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/concpatog/cp_fig5_qqplot.png', dpi=150); plt.close()

### Diagnósticos numéricos de apoio
print("\nShapiro-Wilk:", stats.shapiro(res))
print("Durbin-Watson:", durbin_watson(res))
print("Breusch-Pagan p-valor:", het_breuschpagan(res, m.model.exog)[1])
print("Assimetria resíduos:", stats.skew(res), " Curtose (excesso):", stats.kurtosis(res))
infl = m.get_influence()
d = pd.DataFrame({'ETE': df.ETE, 'Trat': df.Tratamento, 'CPatog1': df.CPatog1, 'CPatog2': df.CPatog2,
                  'fit': fit.round(0), 'resid': res.round(0),
                  'resid_stud': infl.resid_studentized_external.round(2),
                  'alavanca': infl.hat_matrix_diag.round(2)})
print(d.sort_values('resid', key=abs, ascending=False).head(6).to_string())
print("Fração de preditos negativos:", (fit < 0).mean())
print("Fração dos dados com CPatog2 > 1e5:", (df.CPatog2 > 1e5).mean())