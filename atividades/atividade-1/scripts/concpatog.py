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
                  'alavanca': infl.hat_matrix_diag.round(2), 'cook': infl.cooks_distance[0].round(2)})
print(d.sort_values('resid', key=abs, ascending=False).head(6).to_string())
print("Ponto de maior influência:\n", d.loc[[d.cook.idxmax()]].to_string())
print("Correlação de Pearson:", np.corrcoef(df.CPatog1, df.CPatog2)[0, 1],
      "| Spearman (postos):", stats.spearmanr(df.CPatog1, df.CPatog2).statistic)
ms1 = smf.ols('CPatog2 ~ CPatog1', data=df.drop(index=d.cook.idxmax())).fit()
print(f"Sem o ponto de maior influência: b1={ms1.params['CPatog1']:.2f}  R2aj={ms1.rsquared_adj:.4f}")
print("Fração com CPatog1 < 2000:", (df.CPatog1 < 2000).mean())
print("Fração de preditos negativos:", (fit < 0).mean())
print("Fração dos dados com CPatog2 > 1e5:", (df.CPatog2 > 1e5).mean())

# ======================= SEÇÃO 3.2 – TRANSFORMAÇÕES DE ESCALA =======================
# (usa `df`, `m`, `n`, `np`, `pd`, `plt`, `smf`, `stats`, `het_breuschpagan` já importados)
#
# Tratamento dos zeros: CPatog1 tem 3 zeros (todos no estágio 3.Reuso, o mais tratado), onde log10 é indefinido.
# São concentrações abaixo do limite de detecção, não dados ausentes: descartá-las eliminaria justamente os
# menores valores e enviesaria o ajuste. Usamos então log10(C + 1) (convenção usual para concentrações/contagens
# com zeros), aplicado às duas variáveis para manter a mesma escala. Como as concentrações vão até ~10^6, o "+1"
# só altera os valores muito pequenos. Ao final há uma análise de sensibilidade sem os zeros.
import os
OUT = 'atividades/atividade-1/outputs/concpatog/'
os.makedirs(OUT, exist_ok=True)

# Tabela OLS do modelo original (para o relatório)
with open(OUT + 'cp_ols_original.txt', 'w', encoding='utf-8') as f:
    f.write(m.summary().as_text())

# As 9 possibilidades = 3 transformações x 3 alvos (somente X, somente Y, ambas)
FUNCS  = {'raiz quadrada': np.sqrt, 'quadrado': np.square, 'log10': lambda v: np.log10(v + 1)}
SIMB   = {'raiz quadrada': lambda v: f'√({v})', 'quadrado': lambda v: f'({v})²', 'log10': lambda v: f'log10({v}+1)'}
ALVOS  = ['somente X', 'somente Y', 'X e Y']

def transformar(df, tipo, alvo):
    """Devolve (x, y, rótulo_x, rótulo_y) já transformados."""
    f = FUNCS[tipo]
    x, y = df.CPatog1.astype(float).copy(), df.CPatog2.astype(float).copy()
    lx, ly = 'CPatog1', 'CPatog2'
    if alvo in ('somente X', 'X e Y'):
        x = f(x); lx = SIMB[tipo]('CPatog1')
    if alvo in ('somente Y', 'X e Y'):
        y = f(y); ly = SIMB[tipo]('CPatog2')
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
        ax.scatter(d.x, d.y, s=30, alpha=0.7, color='steelblue', edgecolors='white', zorder=3)
        xs = np.linspace(d.x.min(), d.x.max(), 50)
        ax.plot(xs, mm.params['Intercept'] + mm.params['x'] * xs, color='crimson', lw=1.8)
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.set_title(f'{tipo} – {alvo}  (R²aj={mm.rsquared_adj:.3f})', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'cp_transf_dispersoes.png', dpi=150); plt.close()

# --- (b) Grade 3x3 de resíduos vs. preditos (apoio: curvatura e funil são mais fáceis de ver aqui)
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
for i, tipo in enumerate(FUNCS):
    for j, alvo in enumerate(ALVOS):
        x, y, lx, ly = transformar(df, tipo, alvo)
        mm = smf.ols('y ~ x', data=pd.DataFrame({'x': x, 'y': y})).fit()
        ax = axs[i, j]
        ax.scatter(mm.fittedvalues, mm.resid, s=30, alpha=0.7, color='steelblue', edgecolors='white', zorder=3)
        ax.axhline(0, color='crimson', ls='--')
        ax.set_xlabel('preditos'); ax.set_ylabel('resíduos'); ax.set_title(f'{tipo} – {alvo}', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'cp_transf_residuos.png', dpi=150); plt.close()

# --- (c) Tabela-resumo (referência: dados originais)
z0 = (df.CPatog1 - df.CPatog1.mean()) / df.CPatog1.std()
p_quad0 = smf.ols('CPatog2 ~ z + I(z**2)', data=df.assign(z=z0)).fit().pvalues['I(z ** 2)']
print(f"\nOriginal: n={n}  r={np.corrcoef(df.CPatog1, df.CPatog2)[0,1]:.4f}  R2aj={m.rsquared_adj:.4f}  p_termo_quad={p_quad0:.4g}")
tab = pd.DataFrame(linhas)
print(tab.round(4).to_string(index=False))
tab.to_csv(OUT + 'cp_transf_resumo.csv', index=False)

# ======================= SEÇÃO 3.3 – ANÁLISE COM A COMBINAÇÃO ESCOLHIDA =======================
# Combinação escolhida: log10 aplicado em X e em Y  ->  log10(CPatog2+1) ~ log10(CPatog1+1)
# (R²aj ≈ 0,66 contra no máximo ≈ 0,26 nas demais 8 combinações; resíduos aproximadamente normais e homocedásticos)
XLT = 'log10(CPatog1 + 1)'
YLT = 'log10(CPatog2 + 1)'
C1, C2 = '#2b6cb0', '#c53030'
CORES_TRAT = {'1.Bruto': '#2b6cb0', '2.Secundario': '#dd6b20', '3.Reuso': '#2f855a'}

d3 = df.copy()
d3['log_p1'] = np.log10(d3.CPatog1 + 1)
d3['log_p2'] = np.log10(d3.CPatog2 + 1)

# 1. Dispersão (variáveis transformadas)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(d3.log_p1, d3.log_p2, s=40, alpha=0.8, color=C1, edgecolors='white', zorder=3)
ax.set_xlabel(XLT); ax.set_ylabel(YLT); ax.set_title('Dispersão: log10(CPatog2+1) vs. log10(CPatog1+1)')
plt.tight_layout(); plt.savefig(OUT + 'cp_log_fig1_dispersao.png', dpi=200); plt.close()

# 2. Ajuste
m3 = smf.ols('log_p2 ~ log_p1', data=d3).fit()
print(m3.summary())
with open(OUT + 'cp_ols_log.txt', 'w', encoding='utf-8') as f:
    f.write(m3.summary().as_text())
print("R2 =", m3.rsquared, " R2adj =", m3.rsquared_adj, " RSE =", np.sqrt(m3.mse_resid))
print("Interpretação: CPatog2 ≈ 10^b0 · CPatog1^b1 =", 10**m3.params['Intercept'], "· CPatog1 ^", m3.params['log_p1'])
print("Teste H0: b1 = 1 (proporcionalidade):", m3.t_test('log_p1 = 1'))
print("Amplitude do IP 95%% (em fator multiplicativo): ±%.1fx" % 10**(stats.t.ppf(0.975, m3.df_resid) * np.sqrt(m3.mse_resid)))

# Reta + IC + IP (escala transformada)
xr = pd.DataFrame({'log_p1': np.linspace(d3.log_p1.min(), d3.log_p1.max(), 200)})
sf3 = m3.get_prediction(xr).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.fill_between(xr.log_p1, sf3.obs_ci_lower, sf3.obs_ci_upper, color='gray', alpha=0.18, label='Intervalo de predição 95%')
ax.fill_between(xr.log_p1, sf3.mean_ci_lower, sf3.mean_ci_upper, color=C2, alpha=0.28, label='Intervalo de confiança 95%')
ax.plot(xr.log_p1, sf3['mean'], color=C2, lw=2, label='Reta ajustada')
ax.scatter(d3.log_p1, d3.log_p2, s=40, alpha=0.8, color=C1, edgecolors='white', zorder=3, label='Observações')
ax.set_xlabel(XLT); ax.set_ylabel(YLT); ax.legend(fontsize=9, loc='upper left')
ax.set_title(f'Reta de regressão com IC e IP (R² = {m3.rsquared:.3f})')
plt.tight_layout(); plt.savefig(OUT + 'cp_log_fig2_reta_ic_ip.png', dpi=200); plt.close()

# Resíduos: histograma + resíduos vs. preditos + QQ-plot
res3, fit3 = m3.resid, m3.fittedvalues
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].hist(res3, bins=12, density=True, color=C1, edgecolor='white', alpha=0.85, label='Resíduos')
xs = np.linspace(res3.min() - 0.5, res3.max() + 0.5, 300)
axs[0].plot(xs, stats.norm.pdf(xs, 0, res3.std(ddof=1)), color=C2, lw=2, label='Normal teórica')
axs[0].axvline(0, color='black', ls='--', lw=1)
axs[0].set_xlabel('Resíduos (log10)'); axs[0].set_ylabel('Densidade'); axs[0].legend(fontsize=8)
axs[0].set_title('Histograma dos resíduos')
axs[1].scatter(fit3, res3, s=40, alpha=0.8, color=C1, edgecolors='white', zorder=3); axs[1].axhline(0, color=C2, ls='--')
axs[1].set_xlabel('Valores preditos (log10)'); axs[1].set_ylabel('Resíduos (log10)')
axs[1].set_title('Resíduos vs. valores preditos')
sm.qqplot(res3, line='s', ax=axs[2], markerfacecolor=C1, markeredgecolor='white', markersize=6)
axs[2].get_lines()[1].set_color(C2)
axs[2].set_xlabel('Quantis teóricos (normal)'); axs[2].set_ylabel('Quantis amostrais'); axs[2].set_title('QQ-plot dos resíduos')
plt.tight_layout(); plt.savefig(OUT + 'cp_log_fig3_residuos.png', dpi=200); plt.close()

# Extra (apoio à premissa de independência): mesma dispersão, colorida pelo estágio de tratamento
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.plot(xr.log_p1, sf3['mean'], color='black', lw=1.5, ls='--', label='Reta ajustada')
for trat, cor in CORES_TRAT.items():
    g = d3[d3.Tratamento == trat]
    ax.scatter(g.log_p1, g.log_p2, s=40, alpha=0.85, color=cor, edgecolors='white', zorder=3, label=trat)
ax.set_xlabel(XLT); ax.set_ylabel(YLT); ax.legend(fontsize=9, loc='upper left', title='Tratamento', title_fontsize=9)
ax.set_title('Observações por estágio de tratamento')
plt.tight_layout(); plt.savefig(OUT + 'cp_log_fig4_por_tratamento.png', dpi=200); plt.close()

# Diagnósticos numéricos (apoio ao texto)
print("\nShapiro-Wilk:", stats.shapiro(res3))
print("Durbin-Watson:", durbin_watson(res3))
print("Breusch-Pagan: LM p-valor =", het_breuschpagan(res3, m3.model.exog)[1])
print("Assimetria:", stats.skew(res3), " Curtose (excesso):", stats.kurtosis(res3))
infl3 = m3.get_influence()
d = pd.DataFrame({'ETE': d3.ETE, 'Trat': d3.Tratamento, 'CPatog1': d3.CPatog1, 'CPatog2': d3.CPatog2,
                  'fit': fit3.round(2), 'resid': res3.round(2), 'resid_stud': infl3.resid_studentized_external.round(2),
                  'alavanca': infl3.hat_matrix_diag.round(3), 'cook': infl3.cooks_distance[0].round(3)})
print(d.sort_values('resid', key=abs, ascending=False).head(6).to_string())
print("Pontos com CPatog1 = 0:\n", d[d.CPatog1 == 0].to_string())
print("Resíduos por estágio de tratamento:\n", d3.assign(res=res3).groupby('Tratamento').res.agg(['mean', 'std', 'count']))
print("Resíduos por ETE:\n", d3.assign(res=res3).groupby('ETE').res.agg(['mean', 'std', 'count']))
# Independência: os resíduos dependem dos fatores de agrupamento? (ANOVA de um fator sobre os resíduos)
from statsmodels.stats.anova import anova_lm
for fator in ['Tratamento', 'ETE', 'Mes']:
    p = anova_lm(smf.ols(f'res ~ C({fator})', data=d3.assign(res=res3)).fit())['PR(>F)'].iloc[0]
    print(f"ANOVA resíduos ~ {fator}: p = {p:.4f}")
print("Resíduo médio por mês:\n", d3.assign(res=res3).groupby('Mes').res.mean().sort_values().round(2).to_string())
print("Resíduos estudentizados com |r| > 2:", int((np.abs(infl3.resid_studentized_external) > 2).sum()), "de", n)
# Quanto da associação existe dentro de cada estágio (e não só pelo gradiente bruto -> secundário -> reuso)?
for trat, g in d3.groupby('Tratamento'):
    mg = smf.ols('log_p2 ~ log_p1', data=g).fit()
    print(f"Dentro de {trat} (n={len(g)}): b1={mg.params['log_p1']:.3f}  p={mg.pvalues['log_p1']:.3g}  R2={mg.rsquared:.3f}")

# Comparação na escala original: como os valores vão de unidades a milhões, usa-se o fator de erro |log10(predito/observado)|
# (o RMSE em concentração seria dominado pelas poucas amostras com valores na casa dos milhões)
for nome, pred in [('original', fit), ('log', 10**fit3 - 1)]:
    fator = np.abs(np.log10(np.clip(pred, 1, None) / df.CPatog2))
    print(f"Modelo {nome}: fator de erro mediano = {10**np.median(fator):.1f}x  "
          f"| fração das amostras previstas dentro de 10x = {(fator <= 1).mean():.0%}")
print("Maior distância de Cook (modelo log):", infl3.cooks_distance[0].max())

# Sensibilidade: mesmo modelo log-log descartando os 3 zeros (log10 puro, sem +1)
ds = df[df.CPatog1 > 0].assign(lx=lambda t: np.log10(t.CPatog1), ly=lambda t: np.log10(t.CPatog2))
ms = smf.ols('ly ~ lx', data=ds).fit()
print(f"\nSensibilidade (sem zeros, n={len(ds)}): b0={ms.params['Intercept']:.4f}  b1={ms.params['lx']:.4f}  "
      f"R2aj={ms.rsquared_adj:.4f}  | modelo escolhido: b0={m3.params['Intercept']:.4f}  b1={m3.params['log_p1']:.4f}  R2aj={m3.rsquared_adj:.4f}")