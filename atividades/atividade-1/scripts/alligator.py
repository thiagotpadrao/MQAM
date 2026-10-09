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
                  'alavanca': infl.hat_matrix_diag.round(2), 'cook': infl.cooks_distance[0].round(2)})
print(d.sort_values('resid', key=abs, ascending=False).head(6))
print("Correlação de Pearson:", np.corrcoef(df.length, df.weight)[0, 1])
print("Comprimento abaixo do qual o peso predito é negativo:", -m.params['Intercept'] / m.params['length'])
print("Sinais dos resíduos (ordenados pelo predito):", np.sign(res[np.argsort(fit.values)]).astype(int).tolist())

# ======================= SEÇÃO 2.2 – TRANSFORMAÇÕES DE ESCALA =======================
# (usa `df`, `m`, `n`, `np`, `pd`, `plt`, `smf`, `stats`, `het_breuschpagan` já importados)
import os
OUT = 'atividades/atividade-1/outputs/alligator/'
os.makedirs(OUT, exist_ok=True)

# Tabela OLS do modelo original (para o relatório)
with open(OUT + 'alg_ols_original.txt', 'w', encoding='utf-8') as f:
    f.write(m.summary().as_text())

# As 9 possibilidades = 3 transformações x 3 alvos (somente X, somente Y, ambas)
FUNCS  = {'raiz quadrada': np.sqrt, 'quadrado': np.square, 'log10': np.log10}
SIMB   = {'raiz quadrada': lambda v: f'√({v})', 'quadrado': lambda v: f'({v})²', 'log10': lambda v: f'log10({v})'}
ALVOS  = ['somente X', 'somente Y', 'X e Y']

def transformar(df, tipo, alvo):
    """Devolve (x, y, rótulo_x, rótulo_y) já transformados (não há zeros: length >= 58, weight >= 28)."""
    f = FUNCS[tipo]
    x, y = df.length.astype(float).copy(), df.weight.astype(float).copy()
    lx, ly = 'length', 'weight'
    if alvo in ('somente X', 'X e Y'):
        x = f(x); lx = SIMB[tipo]('length')
    if alvo in ('somente Y', 'X e Y'):
        y = f(y); ly = SIMB[tipo]('weight')
    return x, y, lx, ly

# --- (a) Grade 3x3 de dispersões (apenas para análise; não entra no relatório)
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
linhas = []
for i, tipo in enumerate(FUNCS):
    for j, alvo in enumerate(ALVOS):
        x, y, lx, ly = transformar(df, tipo, alvo)
        d = pd.DataFrame({'x': x, 'y': y})
        mm = smf.ols('y ~ x', data=d).fit()
        z = (d.x - d.x.mean()) / d.x.std()                       # padroniza x: evita instabilidade numérica em x² (p-valor não muda)
        mq = smf.ols('y ~ z + I(z**2)', data=d.assign(z=z)).fit()  # termo quadrático: indicador de curvatura remanescente
        linhas.append({'transformação': tipo, 'alvo': alvo, 'n': len(d),
                       'r': np.corrcoef(d.x, d.y)[0, 1], 'R2_aj': mm.rsquared_adj,
                       'p_termo_quad': mq.pvalues['I(z ** 2)'],
                       'p_shapiro': stats.shapiro(mm.resid).pvalue,
                       'p_breusch_pagan': het_breuschpagan(mm.resid, mm.model.exog)[1]})
        ax = axs[i, j]
        ax.scatter(d.x, d.y, s=45, color='steelblue', edgecolors='white', zorder=3)
        xs = np.linspace(d.x.min(), d.x.max(), 50)
        ax.plot(xs, mm.params['Intercept'] + mm.params['x'] * xs, color='crimson', lw=1.8)
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.set_title(f'{tipo} – {alvo}  (R²aj={mm.rsquared_adj:.3f})', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'alg_transf_dispersoes.png', dpi=150); plt.close()

# --- (b) Grade 3x3 de resíduos vs. preditos (apoio: curvatura e funil são mais fáceis de ver aqui)
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
for i, tipo in enumerate(FUNCS):
    for j, alvo in enumerate(ALVOS):
        x, y, lx, ly = transformar(df, tipo, alvo)
        mm = smf.ols('y ~ x', data=pd.DataFrame({'x': x, 'y': y})).fit()
        ax = axs[i, j]
        ax.scatter(mm.fittedvalues, mm.resid, s=45, color='steelblue', edgecolors='white', zorder=3)
        ax.axhline(0, color='crimson', ls='--')
        ax.set_xlabel('preditos'); ax.set_ylabel('resíduos'); ax.set_title(f'{tipo} – {alvo}', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'alg_transf_residuos.png', dpi=150); plt.close()

# --- (c) Tabela-resumo (referência: dados originais)
z0 = (df.length - df.length.mean()) / df.length.std()
p_quad0 = smf.ols('weight ~ z + I(z**2)', data=df.assign(z=z0)).fit().pvalues['I(z ** 2)']
print(f"\nOriginal: n={n}  r={np.corrcoef(df.length, df.weight)[0,1]:.4f}  R2aj={m.rsquared_adj:.4f}  p_termo_quad={p_quad0:.4g}")
tab = pd.DataFrame(linhas)
print(tab.round(4).to_string(index=False))
tab.to_csv(OUT + 'alg_transf_resumo.csv', index=False)

# ======================= SEÇÃO 2.3 – ANÁLISE COM A COMBINAÇÃO ESCOLHIDA =======================
# Combinação escolhida: log10 aplicado somente em Y  ->  log10(weight) ~ length
# (única das 9 sem curvatura remanescente: p do termo quadrático ≈ 0,49; log10 em X e Y fica em 2º, p ≈ 0,006)
XL  = 'Comprimento (polegadas)'
YLT = 'log10 do peso (log10 libras)'
C1, C2 = '#2b6cb0', '#c53030'

d3 = df.copy()
d3['log_weight'] = np.log10(d3.weight)

# 1. Dispersão (variáveis transformadas)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(d3.length, d3.log_weight, s=55, color=C1, edgecolors='white', zorder=3)
ax.set_xlabel(XL); ax.set_ylabel(YLT); ax.set_title('Dispersão: log10(weight) vs. length')
plt.tight_layout(); plt.savefig(OUT + 'alg_log_fig1_dispersao.png', dpi=200); plt.close()

# 2. Ajuste
m3 = smf.ols('log_weight ~ length', data=d3).fit()
print(m3.summary())
with open(OUT + 'alg_ols_log.txt', 'w', encoding='utf-8') as f:
    f.write(m3.summary().as_text())
print("R2 =", m3.rsquared, " R2adj =", m3.rsquared_adj, " RSE =", np.sqrt(m3.mse_resid))
print("Interpretação: cada polegada a mais multiplica weight por 10^b1 =", 10**m3.params['length'],
      "(aumento médio de %.2f%%)" % ((10**m3.params['length'] - 1) * 100))
print("IC95%% do fator multiplicativo: [%.4f, %.4f]" % tuple(sorted(10**m3.conf_int().loc['length'].values)))

# Reta + IC + IP (escala transformada)
xr = pd.DataFrame({'length': np.linspace(d3.length.min(), d3.length.max(), 100)})
sf3 = m3.get_prediction(xr).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.fill_between(xr.length, sf3.obs_ci_lower, sf3.obs_ci_upper, color='gray', alpha=0.18, label='Intervalo de predição 95%')
ax.fill_between(xr.length, sf3.mean_ci_lower, sf3.mean_ci_upper, color=C2, alpha=0.28, label='Intervalo de confiança 95%')
ax.plot(xr.length, sf3['mean'], color=C2, lw=2, label='Reta ajustada')
ax.scatter(d3.length, d3.log_weight, s=55, color=C1, edgecolors='white', zorder=3, label='Observações')
ax.set_xlabel(XL); ax.set_ylabel(YLT); ax.legend(fontsize=9, loc='upper left')
ax.set_title(f'Reta de regressão com IC e IP (R² = {m3.rsquared:.3f})')
plt.tight_layout(); plt.savefig(OUT + 'alg_log_fig2_reta_ic_ip.png', dpi=200); plt.close()

# Resíduos: histograma + resíduos vs. preditos + QQ-plot
res3, fit3 = m3.resid, m3.fittedvalues
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].hist(res3, bins=6, density=True, color=C1, edgecolor='white', alpha=0.85, label='Resíduos')
xs = np.linspace(res3.min() - 0.1, res3.max() + 0.1, 300)
axs[0].plot(xs, stats.norm.pdf(xs, 0, res3.std(ddof=1)), color=C2, lw=2, label='Normal teórica')
axs[0].axvline(0, color='black', ls='--', lw=1)
axs[0].set_xlabel('Resíduos (log10 libras)'); axs[0].set_ylabel('Densidade'); axs[0].legend(fontsize=8)
axs[0].set_title('Histograma dos resíduos')
axs[1].scatter(fit3, res3, s=55, color=C1, edgecolors='white', zorder=3); axs[1].axhline(0, color=C2, ls='--')
axs[1].set_xlabel('Valores preditos (log10 libras)'); axs[1].set_ylabel('Resíduos (log10 libras)')
axs[1].set_title('Resíduos vs. valores preditos')
sm.qqplot(res3, line='s', ax=axs[2], markerfacecolor=C1, markeredgecolor='white', markersize=8)
axs[2].get_lines()[1].set_color(C2)
axs[2].set_xlabel('Quantis teóricos (normal)'); axs[2].set_ylabel('Quantis amostrais'); axs[2].set_title('QQ-plot dos resíduos')
plt.tight_layout(); plt.savefig(OUT + 'alg_log_fig3_residuos.png', dpi=200); plt.close()

# Extra (comparação com o modelo original): curva ajustada de volta à escala original (libras)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.fill_between(xr.length, 10**sf3.obs_ci_lower, 10**sf3.obs_ci_upper, color='gray', alpha=0.18, label='Intervalo de predição 95%')
ax.plot(xr.length, 10**sf3['mean'], color=C2, lw=2, label='Modelo log10(weight)')
ax.plot(xr.length, m.params['Intercept'] + m.params['length'] * xr.length, color='black', lw=1.5, ls='--', label='Modelo original (reta)')
ax.scatter(df.length, df.weight, s=55, color=C1, edgecolors='white', zorder=3, label='Observações')
ax.set_xlabel(XL); ax.set_ylabel('Peso (libras)'); ax.legend(fontsize=9, loc='upper left')
ax.set_title('Comparação na escala original')
plt.tight_layout(); plt.savefig(OUT + 'alg_log_fig4_escala_original.png', dpi=200); plt.close()

# Diagnósticos numéricos (apoio ao texto)
print("\nShapiro-Wilk:", stats.shapiro(res3))
print("Durbin-Watson:", durbin_watson(res3))
print("Breusch-Pagan: LM p-valor =", het_breuschpagan(res3, m3.model.exog)[1])
print("Assimetria:", stats.skew(res3), " Curtose (excesso):", stats.kurtosis(res3))
infl3 = m3.get_influence()
d = pd.DataFrame({'length': d3.length, 'weight': d3.weight, 'fit': fit3.round(3), 'resid': res3.round(3),
                  'resid_stud': infl3.resid_studentized_external.round(2), 'alavanca': infl3.hat_matrix_diag.round(2),
                  'cook': infl3.cooks_distance[0].round(3)})
print("Maior distância de Cook (modelo log):", infl3.cooks_distance[0].max())
print(d.sort_values('resid', key=abs, ascending=False).head(6))
print("RMSE na escala original (libras): modelo log =", np.sqrt(np.mean((df.weight - 10**fit3)**2)),
      "| modelo original =", np.sqrt(np.mean(m.resid**2)))
for nome, pred in [('original', fit), ('log', 10**fit3)]:
    erro_rel = np.abs(df.weight - pred) / df.weight
    print(f"Modelo {nome}: MAE={np.mean(np.abs(df.weight - pred)):.1f} libras  erro relativo mediano={erro_rel.median():.1%}  "
          f"maior erro relativo={erro_rel.max():.1%} (length={df.length[erro_rel.idxmax()]})")
print("Sinais dos resíduos (ordenados pelo predito):", np.sign(res3[np.argsort(fit3.values)]).astype(int).tolist())
print("corr(|resíduo|, predito):", np.corrcoef(np.abs(res3), fit3)[0, 1])
print("Inclinação do modelo log10-log10 (alometria):", smf.ols('np.log10(weight) ~ np.log10(length)', data=df).fit().params.iloc[1])