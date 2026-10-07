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

#### Carregamento (arquivo separado por espaços, com espaços no início das linhas)
df = pd.read_csv('atividades/atividade-1/datasets/alligator.txt', sep=r'\s+')
df.columns = df.columns.str.strip()
n = len(df)
print(df.shape); print(df.describe())

# Dispersão
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(df.length, df.weight, s=70, color='steelblue', edgecolors='white')
ax.set_xlabel('length (polegadas)'); ax.set_ylabel('weight (libras)')
ax.set_title('Dispersão: weight vs. length')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/alligator/alg_fig1_dispersao.png', dpi=150); plt.close()

# Ajuste
m = smf.ols('weight ~ length', data=df).fit()
print(m.summary())
print("R2 =", m.rsquared, " R2adj =", m.rsquared_adj)

# Reta + IC + IP
xr = pd.DataFrame({'length': np.linspace(df.length.min(), df.length.max(), 100)})
sf = m.get_prediction(xr).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(df.length, df.weight, s=70, color='steelblue', edgecolors='white', label='Dados', zorder=3)
ax.plot(xr.length, sf['mean'], color='crimson', lw=2.5, label='Reta de regressão')
ax.fill_between(xr.length, sf.mean_ci_lower, sf.mean_ci_upper, color='crimson', alpha=0.25, label='IC 95% (média)')
ax.fill_between(xr.length, sf.obs_ci_lower, sf.obs_ci_upper, color='gray', alpha=0.15, label='IP 95% (nova obs.)')
ax.set_xlabel('length'); ax.set_ylabel('weight'); ax.legend()
ax.set_title(f'Reta de regressão com IC e IP (R² = {m.rsquared:.3f})')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/alligator/alg_fig2_reta_ic_ip.png', dpi=150); plt.close()

#### Resíduos
res = m.resid; fit = m.fittedvalues

# Histograma + densidade normal teórica (média 0, desvio = std dos resíduos)
fig, ax = plt.subplots(figsize=(8,5))
bins = max(6, int(n/4))   # n=25 é pequeno: ~6 classes (ajuste se quiser)
ax.hist(res, bins=bins, density=True, color='steelblue', edgecolor='white', alpha=0.8, label='Resíduos')
xs = np.linspace(res.min()-20, res.max()+20, 300)
ax.plot(xs, stats.norm.pdf(xs, res.mean(), res.std(ddof=1)), color='crimson', lw=2.5, label='Normal teórica')
ax.set_xlabel('Resíduos'); ax.set_ylabel('Densidade'); ax.legend()
ax.set_title('Histograma dos resíduos')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/alligator/alg_fig3_hist_res.png', dpi=150); plt.close()

# Resíduos vs. preditos
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(fit, res, s=70, color='steelblue', edgecolors='white')
ax.axhline(0, color='crimson', ls='--')
ax.set_xlabel('Valores preditos'); ax.set_ylabel('Resíduos'); ax.set_title('Resíduos vs. valores preditos')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/alligator/alg_fig4_res_vs_pred.png', dpi=150); plt.close()

# QQ-plot
fig, ax = plt.subplots(figsize=(6,6))
sm.qqplot(res, line='s', ax=ax, markerfacecolor='steelblue', markeredgecolor='white', markersize=9)
ax.set_title('QQ-plot dos resíduos (normal)')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/alligator/alg_fig5_qqplot.png', dpi=150); plt.close()

### Diagnósticos numéricos (apoio para a discussão LINE; não precisam ir todos ao relatório)
print("\nShapiro-Wilk:", stats.shapiro(res))
print("Durbin-Watson:", durbin_watson(res))
bp = het_breuschpagan(res, m.model.exog)
print("Breusch-Pagan: LM p-valor =", bp[1])
print("Assimetria dos resíduos:", stats.skew(res), " Curtose (excesso):", stats.kurtosis(res))
infl = m.get_influence()
d = pd.DataFrame({'length': df.length, 'weight': df.weight, 'fit': fit.round(1),
                  'resid': res.round(1), 'resid_stud': infl.resid_studentized_external.round(2),
                  'alavanca': infl.hat_matrix_diag.round(2)})
print(d.sort_values('resid', key=abs, ascending=False).head(6))