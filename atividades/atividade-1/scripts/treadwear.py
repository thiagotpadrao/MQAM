import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
 
plt.style.use('seaborn-v0_8-whitegrid')
df = pd.read_csv('atividades/atividade-1/datasets/treadwear.txt', sep='\t')
df.columns = df.columns.str.strip()
print(df.shape); print(df.describe())
 
# Dispersão
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(df.mileage, df.groove, s=70, color='steelblue', edgecolors='white')
ax.set_xlabel('mileage (milhares de milhas)'); ax.set_ylabel('groove (profundidade do sulco, mils)')
ax.set_title('Dispersão: groove vs. mileage')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/treadwear/treadwearfig1_dispersao.png', dpi=150); plt.close()
 
# Ajuste
m = smf.ols('groove ~ mileage', data=df).fit()
print(m.summary())
print("R2 =", m.rsquared, " R2adj =", m.rsquared_adj)
 
# Reta + IC + IP
xr = pd.DataFrame({'mileage': np.linspace(df.mileage.min(), df.mileage.max(), 100)})
sf = m.get_prediction(xr).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(df.mileage, df.groove, s=70, color='steelblue', edgecolors='white', label='Dados', zorder=3)
ax.plot(xr.mileage, sf['mean'], color='crimson', lw=2.5, label='Reta de regressão')
ax.fill_between(xr.mileage, sf.mean_ci_lower, sf.mean_ci_upper, color='crimson', alpha=0.25, label='IC 95% (média)')
ax.fill_between(xr.mileage, sf.obs_ci_lower, sf.obs_ci_upper, color='gray', alpha=0.15, label='IP 95% (nova obs.)')
ax.set_xlabel('mileage'); ax.set_ylabel('groove'); ax.legend()
ax.set_title(f'Reta de regressão com IC e IP (R² = {m.rsquared:.3f})')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/treadwear/fig2_reta_ic_ip.png', dpi=150); plt.close()
 
# Resíduos
res = m.resid; fit = m.fittedvalues
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(fit, res, s=70, color='steelblue', edgecolors='white')
ax.axhline(0, color='crimson', ls='--')
ax.set_xlabel('Valores preditos'); ax.set_ylabel('Resíduos'); ax.set_title('Resíduos vs. valores preditos')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/treadwear/fig3_res_vs_pred.png', dpi=150); plt.close()
 
fig, ax = plt.subplots(figsize=(6,6))
sm.qqplot(res, line='s', ax=ax, markerfacecolor='steelblue', markeredgecolor='white', markersize=9)
ax.set_title('QQ-plot dos resíduos (normal)')
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/treadwear/fig4_qqplot.png', dpi=150); plt.close()
 
# Diagnósticos numéricos
print("\nResíduos:\n", pd.DataFrame({'mileage':df.mileage,'fit':fit.round(2),'resid':res.round(2)}))
print("Shapiro-Wilk:", stats.shapiro(res))
print("Durbin-Watson:", durbin_watson(res))
print("Sinais dos resíduos:", np.sign(res).astype(int).tolist())