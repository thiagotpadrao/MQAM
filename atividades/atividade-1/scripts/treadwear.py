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
plt.tight_layout(); plt.savefig('atividades/atividade-1/outputs/treadwear/fig1_dispersao.png', dpi=150); plt.close()
 
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

# ======================= SEÇÃO 1.2 – TRANSFORMAÇÕES DE ESCALA =======================
# (acrescentar ao final do script; usa `df`, `np`, `pd`, `plt`, `smf`, `durbin_watson` já importados)
import os
OUT = 'atividades/atividade-1/outputs/treadwear/'
os.makedirs(OUT, exist_ok=True)
 
# As 9 possibilidades = 3 transformações x 3 alvos (somente X, somente Y, ambas)
FUNCS  = {'raiz quadrada': np.sqrt, 'quadrado': np.square, 'log10': np.log10}
SIMB   = {'raiz quadrada': lambda v: f'√({v})', 'quadrado': lambda v: f'({v})²', 'log10': lambda v: f'log10({v})'}
ALVOS  = ['somente X', 'somente Y', 'X e Y']
 
def transformar(df, tipo, alvo):
    """Devolve (x, y, rótulo_x, rótulo_y) já transformados. Linhas inválidas (ex.: log10(0)) viram NaN."""
    f = FUNCS[tipo]
    x, y = df.mileage.astype(float).copy(), df.groove.astype(float).copy()
    lx, ly = 'mileage', 'groove'
    with np.errstate(divide='ignore', invalid='ignore'):
        if alvo in ('somente X', 'X e Y'):
            x = f(x); lx = SIMB[tipo]('mileage')
        if alvo in ('somente Y', 'X e Y'):
            y = f(y); ly = SIMB[tipo]('groove')
    x = x.replace([np.inf, -np.inf], np.nan); y = y.replace([np.inf, -np.inf], np.nan)
    return x, y, lx, ly
 
# --- (a) Grade 3x3 de dispersões (apenas para análise; não entra no relatório)
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
linhas = []
for i, tipo in enumerate(FUNCS):
    for j, alvo in enumerate(ALVOS):
        x, y, lx, ly = transformar(df, tipo, alvo)
        ok = x.notna() & y.notna()
        d = pd.DataFrame({'x': x[ok], 'y': y[ok]})
        mm = smf.ols('y ~ x', data=d).fit()
        mq = smf.ols('y ~ x + I(x**2)', data=d).fit()          # termo quadrático: indicador de curvatura remanescente
        linhas.append({'transformação': tipo, 'alvo': alvo, 'n': int(ok.sum()),
                       'r': np.corrcoef(d.x, d.y)[0, 1], 'R2_aj': mm.rsquared_adj,
                       'DW': durbin_watson(mm.resid), 'p_termo_quad': mq.pvalues['I(x ** 2)']})
        ax = axs[i, j]
        ax.scatter(d.x, d.y, s=45, color='steelblue', edgecolors='white', zorder=3)
        xs = np.linspace(d.x.min(), d.x.max(), 50)
        ax.plot(xs, mm.params['Intercept'] + mm.params['x'] * xs, color='crimson', lw=1.8)
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.set_title(f'{tipo} – {alvo}  (n={int(ok.sum())}, R²aj={mm.rsquared_adj:.3f})', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'treadwear_transf_dispersoes.png', dpi=150); plt.close()
 
# --- (b) Grade 3x3 de resíduos vs. preditos (apoio: curvatura é mais fácil de ver aqui)
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
for i, tipo in enumerate(FUNCS):
    for j, alvo in enumerate(ALVOS):
        x, y, lx, ly = transformar(df, tipo, alvo)
        ok = x.notna() & y.notna()
        d = pd.DataFrame({'x': x[ok], 'y': y[ok]})
        mm = smf.ols('y ~ x', data=d).fit()
        ax = axs[i, j]
        ax.scatter(mm.fittedvalues, mm.resid, s=45, color='steelblue', edgecolors='white', zorder=3)
        ax.axhline(0, color='crimson', ls='--')
        ax.set_xlabel('preditos'); ax.set_ylabel('resíduos'); ax.set_title(f'{tipo} – {alvo}', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'treadwear_transf_residuos.png', dpi=150); plt.close()
 
# --- (c) Tabela-resumo (referência: dados originais)
orig = smf.ols('groove ~ mileage', data=df).fit()
print(f"Original: n={len(df)}  r={np.corrcoef(df.mileage, df.groove)[0,1]:.4f}  R2aj={orig.rsquared_adj:.4f}  DW={durbin_watson(orig.resid):.3f}")
tab = pd.DataFrame(linhas)
print("\nZeros em mileage (log10 indefinido):", int((df.mileage <= 0).sum()))
print(tab.round(4).to_string(index=False))
tab.to_csv(OUT + 'treadwear_transf_resumo.csv', index=False)