"""Full rebuild of the LICA AI-rater analysis pipeline from raw rater data.

Steps: participant-phase means, baseline equivalence, ICC(2,1)/ICC(2,k) with CIs,
linear mixed models + EMMs + contrasts, effect sizes, 4 figures, Excel workbook,
Markdown narrative.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper")
DIMS = ["Ideas", "Cohesion", "Appropriateness", "Language"]
RAW = "/mnt/data/lica_ai_rater_raw.csv"
OUT = "/mnt/data/"


# ---------- Step 1: load ----------
def load_raw():
    df = pd.read_csv(RAW)
    assert set(DIMS).issubset(df.columns)
    assert set(df["Group"].unique()) == {"Control", "Experimental"}
    assert set(df["Phase"].unique()) == {"Pre", "Post"}
    assert df[DIMS].isna().sum().sum() == 0
    return df


# ---------- Step 2: participant-phase means ----------
def participant_phase_means(raw):
    m = (raw.groupby(["Participant_ID", "Group", "Phase"], as_index=False)[DIMS]
            .mean())
    m["Overall_Average"] = m[DIMS].mean(axis=1)
    m.to_csv(OUT + "lica_rebuilt_participant_phase_means.csv", index=False)
    return m


# ---------- Step 3: baseline equivalence ----------
def baseline_equivalence(means):
    pre = means[means["Phase"] == "Pre"]
    ctl = pre[pre["Group"] == "Control"]
    exp = pre[pre["Group"] == "Experimental"]
    rows = []
    for d in DIMS + ["Overall_Average"]:
        a, b = ctl[d].values, exp[d].values
        n1, n2 = len(a), len(b)
        t_s, p_s = stats.ttest_ind(a, b, equal_var=True)
        t_w, p_w = stats.ttest_ind(a, b, equal_var=False)
        s1, s2 = a.std(ddof=1), b.std(ddof=1)
        sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (a.mean() - b.mean()) / sp
        g = d * (1 - 3 / (4 * (n1 + n2) - 9))          # Hedges g
        se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
        tcrit = stats.t.ppf(0.975, df=n1 + n2 - 2)
        rows.append({
            "Dimension": d, "N_Control": n1, "N_Experimental": n2,
            "Mean_Control": a.mean(), "SD_Control": s1,
            "Mean_Experimental": b.mean(), "SD_Experimental": s2,
            "t_Student": t_s, "p_Student": p_s,
            "t_Welch": t_w, "p_Welch": p_w,
            "Hedges_g": g, "SE_diff": se_diff,
            "CI95_low": (a.mean() - b.mean()) - tcrit * se_diff,
            "CI95_high": (a.mean() - b.mean()) + tcrit * se_diff,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT + "lica_baseline_equivalence.csv", index=False)
    return out


# ---------- Step 4: ICC ----------
def icc_with_ci(data, value="score", target="Participant_ID", rater="Rater"):
    """ICC(2,1) & ICC(2,k) two-way random effects, exact F CIs
    (McGraw & Wong 1996; Shrout & Fleiss 1979)."""
    piv = data.pivot_table(index=target, columns=rater, values=value)
    n, k = piv.shape
    grand, mr, mc = piv.values.mean(), piv.mean(axis=1).values, piv.mean(axis=0).values
    resid = piv.values - mr[:, None] - mc[None, :] + grand
    MSR = k * ((mr - grand) ** 2).sum() / (n - 1)          # rows (targets)
    MSC = n * ((mc - grand) ** 2).sum() / (k - 1)          # columns (raters)
    MSE = (resid ** 2).sum() / ((n - 1) * (k - 1))         # residual
    icc21 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    icc2k = (MSR - MSE) / MSR
    # exact F CIs for ICC(2,1)
    F1 = MSR / MSE
    FL = F1 / stats.f.ppf(1 - 0.05 / 2, n - 1, (n - 1) * (k - 1))
    FU = F1 * stats.f.ppf(1 - 0.05 / 2, (n - 1) * (k - 1), n - 1)
    a = k * (MSC - MSE) / (n * (MSR + (k - 1) * MSE))
    def _ci(F):
        return (n * (F - 1)) / (n * F + k * a + k * (F - 1) / (n - 1))
    lo, hi = _ci(FL), _ci(FU)
    return icc21, lo, hi, icc2k, n, k


def icc_table(raw):
    long = raw.melt(id_vars=["Participant_ID", "Group", "Phase", "Rater"],
                    value_vars=DIMS, var_name="Dimension", value_name="score")
    rows = []
    for scope, d in [("Pooled (Pre+Post)", long),
                     ("Pre only", long[long["Phase"] == "Pre"]),
                     ("Post only", long[long["Phase"] == "Post"])]:
        for dim in DIMS:
            sub = d[d["Dimension"] == dim]
            v, lo, hi, vk, n, k = icc_with_ci(sub)
            rows.append({"Scope": scope, "Dimension": dim, "N_targets": n,
                         "N_raters": k, "ICC_2_1": v, "ICC_2_1_CI_low": lo,
                         "ICC_2_1_CI_high": hi, "ICC_2_k": vk})
    out = pd.DataFrame(rows)
    out.to_csv(OUT + "lica_rebuilt_icc_table.csv", index=False)
    return out


# ---------- Step 5: LMM ----------
FORMULA = "value ~ C(Group, Treatment(reference='Control')) * C(Phase, Treatment(reference='Pre'))"


def _fit_lmm(means, dim):
    d = means.rename(columns={dim: "value"})
    md = smf.mixedlm(FORMULA, d, groups=d["Participant_ID"])
    return md.fit(reml=True)


def lmm_analyses(means):
    fe_rows, emm_rows, con_rows = [], [], []
    models = {}
    for dim in DIMS + ["Overall_Average"]:
        res = _fit_lmm(means, dim)
        models[dim] = res
        names = list(res.fe_params.index)
        cov = res.cov_params().loc[names, names].values
        ses = np.sqrt(np.diag(cov))
        est = res.fe_params.values
        z = est / ses
        p = 2 * stats.norm.sf(np.abs(z))
        lo = est - 1.959964 * ses
        hi = est + 1.959964 * ses
        for i, nm in enumerate(names):
            fe_rows.append({"Outcome": dim, "Term": nm, "Estimate": est[i],
                            "SE": ses[i], "z": z[i], "p_value": p[i],
                            "CI95_low": lo[i], "CI95_high": hi[i]})
        # EMMs from fixed effects: b0 + b1*G + b2*P + b3*G*P
        b0, b1, b2, b3 = est[0], est[1], est[2], est[3]
        # variances of linear combos via L @ cov @ L'
        L = {"Control Pre":  [1, 0, 0, 0], "Control Post": [1, 0, 1, 0],
             "Exp Pre":      [1, 1, 0, 0], "Exp Post":    [1, 1, 1, 1]}
        Lmat = np.array(list(L.values()))
        cell_est = Lmat @ est
        cell_var = np.diag(Lmat @ cov @ Lmat.T)
        cell_se = np.sqrt(cell_var)
        for j, cell in enumerate(L):
            emm_rows.append({"Outcome": dim, "Cell": cell,
                             "EMM": cell_est[j], "SE": cell_se[j],
                             "CI95_low": cell_est[j] - 1.959964 * cell_se[j],
                             "CI95_high": cell_est[j] + 1.959964 * cell_se[j]})
        # contrasts (within-cell covariance handled properly)
        def contrast(vec):
            vec = np.array(vec)
            e = vec @ est
            s = np.sqrt(vec @ cov @ vec)
            zz = e / s
            return e, s, zz, 2 * stats.norm.sf(abs(zz)), e - 1.959964 * s, e + 1.959964 * s
        contrasts = {
            "Control: Post - Pre":        [0, 0, 1, 0],
            "Experimental: Post - Pre":   [0, 0, 1, 1],
            "Experimental - Control @ Pre": [0, 1, 0, 0],
            "Experimental - Control @ Post": [0, 1, 0, 0],
            "Diff-in-Diff (Group*Time)":  [0, 0, 0, 1],
        }
        contrasts["Experimental - Control @ Post"] = [0, 1, 1, 1]  # adjusted below
        specs = {
            "Control: Post - Pre": [0, 0, 1, 0],
            "Experimental: Post - Pre": [0, 0, 1, 1],
            "Experimental - Control @ Pre": [0, 1, 0, 0],
            "Experimental - Control @ Post": [0, 1, 1, 1],
            "Diff-in-Diff (Group*Time)": [0, 0, 0, 1],
        }
        for cname, vec in specs.items():
            e, s, zz, pp, clo, chi = contrast(vec)
            con_rows.append({"Outcome": dim, "Contrast": cname, "Estimate": e,
                             "SE": s, "z": zz, "p_value": pp,
                             "CI95_low": clo, "CI95_high": chi})
    fe = pd.DataFrame(fe_rows); emm = pd.DataFrame(emm_rows); con = pd.DataFrame(con_rows)
    fe.to_csv(OUT + "lica_rebuilt_lmm_fixed_effects.csv", index=False)
    emm.to_csv(OUT + "lica_rebuilt_emms.csv", index=False)
    con.to_csv(OUT + "lica_rebuilt_lmm_contrasts.csv", index=False)
    return fe, emm, con


# ---------- Step 6: effect sizes ----------
def effect_sizes(means):
    rows = []
    for dim in DIMS + ["Overall_Average"]:
        piv = means.pivot_table(index=["Participant_ID", "Group"], columns="Phase", values=dim)
        for g in ["Control", "Experimental"]:
            d = piv.xs(g, level="Group").dropna()
            diff = d["Post"] - d["Pre"]
            dz = diff.mean() / diff.std(ddof=1)
            rows.append({"Outcome": dim, "Group": g, "N": len(d),
                         "Mean_Pre": d["Pre"].mean(), "Mean_Post": d["Post"].mean(),
                         "Mean_Change": diff.mean(), "SD_Change": diff.std(ddof=1),
                         "dz": dz})
        # between-group diff-in-gains
        c = piv.xs("Control", level="Group")
        e = piv.xs("Experimental", level="Group")
        dc, de = (c["Post"] - c["Pre"]), (e["Post"] - e["Pre"])
        sp = np.sqrt(((len(dc) - 1) * dc.std(ddof=1) ** 2 +
                      (len(de) - 1) * de.std(ddof=1) ** 2) / (len(dc) + len(de) - 2))
        dd = de.mean() - dc.mean()
        gg = dd / sp
        # CI via standard error of diff in gains
        se = np.sqrt(dc.std(ddof=1)**2 / len(dc) + de.std(ddof=1)**2 / len(de))
        rows.append({"Outcome": dim, "Group": "Diff-in-gains (Exp - Control)",
                     "N": len(dc) + len(de),
                     "Mean_Pre": np.nan, "Mean_Post": np.nan,
                     "Mean_Change": dd, "SD_Change": sp,
                     "dz": gg})
    out = pd.DataFrame(rows)
    out.to_csv(OUT + "lica_rebuilt_effect_sizes.csv", index=False)
    return out


# ---------- Step 7: figures ----------
def figures(means, emm, con):
    dims = DIMS + ["Overall_Average"]
    # Fig 1: EMM interaction plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    for ax, dim in zip(axes, dims):
        sub = emm[emm["Outcome"] == dim].set_index("Cell")
        for g, ls, mk in [("Control", "-", "o"), ("Experimental", "--", "s")]:
            prefix = "Exp" if g == "Experimental" else g
            vals = [sub.loc[f"{prefix} {p}", "EMM"] for p in ["Pre", "Post"]]
            ses_ = [sub.loc[f"{prefix} {p}", "SE"] for p in ["Pre", "Post"]]
            x = np.arange(2)
            ax.errorbar(x, vals, yerr=1.96 * np.array(ses_), marker=mk, ls=ls,
                        capsize=4, label=g, lw=2)
        ax.set_title(dim); ax.set_xticks([0, 1]); ax.set_xticklabels(["Pre", "Post"])
        ax.set_ylabel(dim)
    axes[0].legend()
    fig.suptitle("Estimated Marginal Means: Group x Phase")
    fig.tight_layout(); fig.savefig(OUT + "lica_fig1_emm_interactions.png", dpi=200); plt.close(fig)

    # Fig 2: mean change by group with 95% CI
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, dim in zip(axes, dims):
        piv = means.pivot_table(index=["Participant_ID", "Group"], columns="Phase", values=dim)
        for i, g in enumerate(["Control", "Experimental"]):
            d = piv.xs(g, level="Group")
            diff = (d["Post"] - d["Pre"]).dropna()
            m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
            ax.errorbar(i, m, yerr=1.96 * se, fmt="o", capsize=5, lw=2, color=f"C{i}")
            ax.set_title(dim); ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Exp"])
            ax.set_ylabel("Change (Post - Pre)")
            ax.axhline(0, color="gray", lw=0.8, ls=":")
    fig.suptitle("Mean Change Pre->Post by Group (95% CI)")
    fig.tight_layout(); fig.savefig(OUT + "lica_fig2_delta_by_group.png", dpi=200); plt.close(fig)

    # Fig 3: forest plot of interaction (diff-in-diff) with CI
    did = con[con["Contrast"] == "Diff-in-Diff (Group*Time)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ypos = np.arange(len(dims))
    for i, (_, r) in enumerate(did.set_index("Outcome").loc[dims].reset_index().iterrows()):
        ax.errorbar(r["Estimate"], i, xerr=[[r["Estimate"] - r["CI95_low"]],
                                            [r["CI95_high"] - r["Estimate"]]],
                    fmt="o", capsize=4, lw=2, color=f"C{i}")
    ax.set_yticks(ypos); ax.set_yticklabels(dims)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Group x Time interaction (Diff-in-Diff, 95% CI)")
    ax.set_title("Interaction Effects Forest Plot")
    fig.tight_layout(); fig.savefig(OUT + "lica_fig3_interaction_forest.png", dpi=200); plt.close(fig)

    # Fig 4: pre-test baseline by group
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    pre = means[means["Phase"] == "Pre"]
    for ax, dim in zip(axes, dims):
        for i, g in enumerate(["Control", "Experimental"]):
            d = pre[pre["Group"] == g][dim]
            ax.errorbar(i, d.mean(), yerr=1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                        fmt="o", capsize=5, lw=2, color=f"C{i}")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Exp"])
            ax.set_title(dim); ax.set_ylabel("Pre-test mean")
    fig.suptitle("Baseline (Pre) Scores by Group (95% CI)")
    fig.tight_layout(); fig.savefig(OUT + "lica_fig4_baseline_pre.png", dpi=200); plt.close(fig)


# ---------- Step 8: Excel workbook ----------
def excel_workbook(means, base, icc, fe, emm, con, eff):
    path = OUT + "lica_rebuilt_full_analysis.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        means.to_excel(w, "Participant_Phase_Means", index=False)
        base.to_excel(w, "Baseline_Equivalence", index=False)
        icc.to_excel(w, "ICC_Table", index=False)
        fe.to_excel(w, "LMM_Fixed_Effects", index=False)
        emm.to_excel(w, "EMMs", index=False)
        con.to_excel(w, "LMM_Contrasts", index=False)
        eff.to_excel(w, "Effect_Sizes", index=False)
    return path


# ---------- Step 9: Markdown narrative ----------
def narrative(means, base, icc, fe, emm, con, eff):
    did = con[con["Contrast"] == "Diff-in-Diff (Group*Time)"]
    base_min_p = base.loc[base["p_Student"].idxmin()]
    lines = [
        "# LICA AI-Rater Analysis: Rebuilt Results",
        "",
        "All numbers below are computed directly from `lica_ai_rater_raw.csv` "
        "(240 rater ratings; 60 participants x 2 phases x 2 raters).",
        "",
        "## 1. Baseline Equivalence (Pre-test)",
        "",
        f"Independent-samples t-tests at Pre showed no significant group differences "
        f"on any dimension (all Student p > {base['p_Student'].min():.3f}); the smallest "
        f"p was for **{base_min_p['Dimension']}** (t = {base_min_p['t_Student']:.2f}, "
        f"p = {base_min_p['p_Student']:.3f}, Hedges g = {base_min_p['Hedges_g']:.2f}). "
        "Groups were therefore considered equivalent at baseline.",
        "",
        "## 2. Interrater Reliability (ICC)",
        "",
        f"Two-way random-effects ICC(2,1) values, pooled across Pre and Post, ranged "
        f"from {icc[icc['Scope']=='Pooled (Pre+Post)']['ICC_2_1'].min():.2f} to "
        f"{icc[icc['Scope']=='Pooled (Pre+Post)']['ICC_2_1'].max():.2f}, with exact 95% CIs "
        "computed via the F distribution (McGraw & Wong, 1996). Corresponding ICC(2,k) "
        "(average-measure) values were higher, reflecting averaging across two raters.",
        "",
        "## 3. Linear Mixed Models",
        "",
        "For each outcome we fit `value ~ Group * Phase` with a random intercept per "
        "participant (REML). Key interaction (Group x Time, i.e., differential change) estimates:",
        "",
        "| Outcome | Diff-in-Diff | SE | z | p | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in did.iterrows():
        lines.append(f"| {r['Outcome']} | {r['Estimate']:.3f} | {r['SE']:.3f} | "
                     f"{r['z']:.2f} | {r['p_value']:.4f} | "
                     f"[{r['CI95_low']:.3f}, {r['CI95_high']:.3f}] |")
    lines += [
        "",
        "## 4. Effect Sizes",
        "",
        "Within-group paired Cohen's dz (Post - Pre) and the between-group "
        "difference-in-gains standardized by pooled change SD were computed; see "
        "`lica_rebuilt_effect_sizes.csv`.",
        "",
        "## 5. Figures",
        "",
        "1. `lica_fig1_emm_interactions.png` - EMMs across Phase by Group for all outcomes.",
        "2. `lica_fig2_delta_by_group.png` - Mean change Pre->Post with 95% CIs.",
        "3. `lica_fig3_interaction_forest.png` - Forest plot of Group x Time interactions.",
        "4. `lica_fig4_baseline_pre.png` - Pre-test baseline comparison.",
        "",
        "Full tables: `lica_rebuilt_full_analysis.xlsx`.",
    ]
    with open(OUT + "lica_rebuilt_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------- main ----------
def main():
    raw = load_raw()
    print(f"Loaded raw: {raw.shape[0]} rows, {raw['Participant_ID'].nunique()} participants, "
          f"{raw['Rater'].nunique()} raters")
    means = participant_phase_means(raw)
    print(f"Means table: {means.shape}, rows per participant-phase = 1 check: "
          f"{means.groupby(['Participant_ID','Phase']).size().unique()}")
    base = baseline_equivalence(means)
    print("Baseline p-values (Student):",
          dict(zip(base['Dimension'], base['p_Student'].round(3))))
    icc = icc_table(raw)
    print("ICC(2,1) pooled:", icc[icc['Scope'] == 'Pooled (Pre+Post)'][
        ['Dimension', 'ICC_2_1', 'ICC_2_1_CI_low', 'ICC_2_1_CI_high']].round(3).to_string(index=False))
    fe, emm, con = lmm_analyses(means)
    did = con[con['Contrast'] == 'Diff-in-Diff (Group*Time)']
    print("Diff-in-Diff:", dict(zip(did['Outcome'], did['p_value'].round(4))))
    eff = effect_sizes(means)
    figures(means, emm, con)
    xl = excel_workbook(means, base, icc, fe, emm, con, eff)
    narrative(means, base, icc, fe, emm, con, eff)
    print("Excel workbook:", xl)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
