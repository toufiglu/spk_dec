
import importlib
import helpers
importlib.reload(helpers)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# below is the visualization for each vowel for each formant: migrate to another file here
formants = ['f1', 'f2', 'f3']
vowels = sorted(gammatone_longform['phone'].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, formant in zip(axes, formants):
    for v in vowels:
        subset = gammatone_longform.loc[gammatone_longform['phone'] == v, formant]
        ax.hist(subset, bins=30, alpha=0.5, label=v, density=True)
    ax.set_title(formant.upper())
    ax.set_xlabel("Hz")
    ax.legend(fontsize=8, ncol=2)

fig.suptitle("Formant Distributions by Vowel")
plt.tight_layout()
plt.show()
fig.savefig("./figures/formant_distributions/formant_distributions_by_vowel.png", dpi=300, bbox_inches="tight")


for v in vowels:
    # one row × 3 columns for F1/F2/F3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # plot each formant distribution for this vowel
    for ax, formant in zip(axes, formants):
        subset = gammatone_longform.loc[gammatone_longform['phone'] == v, formant]
        ax.hist(subset, bins=30, color="steelblue", alpha=0.6, density=True)
        ax.set_title(formant.upper())
        ax.set_xlabel("Hz")
        ax.set_ylabel("Count")

    # overall title and layout
    fig.suptitle(f"Formant distributions for vowel {v}", fontsize=14)
    plt.tight_layout()

    # save and close
    out_path = f"./figures/formant_distributions/{v}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

#######################################################################
# compute the lda distribution for each layer
layer_stats = {}
vowel_layer_stats = {}
for layer_idx in range(1, num_layers):
    df = pd.read_pickle(f"./wav_2_vec_hidden_lda/lda_acts_{layer_idx}.pkl")

    components = ["lda_1", "lda_2", "lda_3"]
    # guard against missing columns
    missing = [c for c in components if c not in df.columns]
    if missing:
        print(f"Layer {layer_idx}: missing columns {missing}, skipping.")
        continue

    vowels = sorted(df["phone"].dropna().unique())

    # ensure output dir exists
    outdir = f"./figures/lda_distributions/layer_{layer_idx}"
    os.makedirs(outdir, exist_ok=True)

    # plot in one figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, comp in zip(axes, components):
        for v in vowels:
            subset = df.loc[df["phone"] == v, comp].dropna()
            ax.hist(subset, bins=30, alpha=0.5, label=v, density=True)
        ax.set_title(comp.upper())
        ax.set_xlabel("unit")
    # put a single legend on the last axis (avoids duplicate legends)
    axes[-1].legend(fontsize=8, ncol=2, title="vowel")

    fig.suptitle(f"LDA component distributions by vowel — Layer {layer_idx}")
    plt.tight_layout()
    fig.savefig(f"{outdir}/component_distributions_by_vowel.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # create plots one for each vowel
    for v in vowels:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        for ax, component in zip(axes, components):
            subset = df.loc[df['phone'] == v, component].dropna()
            ax.hist(subset, bins=30, alpha=0.6, density=True)  # no color so no mpl style issues
            ax.set_title(component.upper())
            ax.set_xlabel("unit")
            ax.set_ylabel("Count")

        fig.suptitle(f"Component distributions for vowel {v}, layer {layer_idx}", fontsize=14)
        plt.tight_layout()

        out_path = f"{outdir}/vowel_{v}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # compute 
    model = df[["lda_1", "lda_2", "lda_3"]].to_numpy()
    model_mean = model.mean(axis=0)
    model_var = np.cov(model, rowvar=False)
    layer_stats[layer_idx] = (model_mean, model_var)

    vowel_layer_stats[layer_idx] = {}
    for v in vowels:
        vdf = df[df["phone"] == v][["lda_1", "lda_2", "lda_3"]].to_numpy()
        v_mean = vdf.mean(axis=0)
        v_cov  = np.cov(vdf, rowvar=False)
        vowel_layer_stats[layer_idx][v] = (v_mean, v_cov)

lda_1_mean_across_layers = {layer: layer_stats[layer][0][0] for layer in layer_stats}
lda_2_mean_across_layers = {layer: layer_stats[layer][0][1] for layer in layer_stats}
lda_3_mean_across_layers = {layer: layer_stats[layer][0][2] for layer in layer_stats}

lda_1_var_across_layers = {layer: layer_stats[layer][1][0][0] for layer in layer_stats}
lda_2_var_across_layers = {layer: layer_stats[layer][1][1][1] for layer in layer_stats}
lda_3_var_across_layers = {layer: layer_stats[layer][1][2][2] for layer in layer_stats}


# dicts you already created
mean_dicts = {
    "lda_1_mean": lda_1_mean_across_layers,
    "lda_2_mean": lda_2_mean_across_layers,
    "lda_3_mean": lda_3_mean_across_layers,
}

var_dicts = {
    "lda_1_var": lda_1_var_across_layers,
    "lda_2_var": lda_2_var_across_layers,
    "lda_3_var": lda_3_var_across_layers,
}

# ---- Plot means ----
plt.figure(figsize=(7,5))
for name, d in mean_dicts.items():
    layers = sorted(d.keys())
    values = [float(d[l]) for l in layers]
    plt.plot(layers, values, marker='o', label=name)

plt.title("Mean of LDA Components Across Layers")
plt.xlabel("Layer")
plt.ylabel("Mean Value")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# ---- Plot variances ----
plt.figure(figsize=(7,5))
for name, d in var_dicts.items():
    layers = sorted(d.keys())
    values = [float(d[l]) for l in layers]
    plt.plot(layers, values, marker='o', label=name)

plt.title("Variance of LDA Components Across Layers")
plt.xlabel("Layer")
plt.ylabel("Variance")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#######################################################################

summary_dir = "./figures/lda_distributions"
mean_dir = f"{summary_dir}/mean_across_layer"
var_dir = f"{summary_dir}/var_across_layer"


vowels_all = sorted(df["phone"].dropna().unique())
components = ["lda_1","lda_2","lda_3"]

# per vowel trajectories
vowel_mean_dicts = {v: {c: [] for c in components} for v in vowels_all}
vowel_var_dicts  = {v: {c: [] for c in components} for v in vowels_all}
layers = sorted(vowel_layer_stats.keys())

for layer in layers:
    for v in vowels_all:
        mean, cov = vowel_layer_stats[layer][v]
        for i, c in enumerate(components):
            vowel_mean_dicts[v][c].append(float(mean[i]))
            vowel_var_dicts[v][c].append(float(cov[i, i]))

for v in vowels_all:
    plt.figure(figsize=(7,5))
    for i, c in enumerate(components):
        plt.plot(layers, vowel_mean_dicts[v][c], marker='o', label=c)
    plt.title(f"Mean LDA components across layers — vowel {v}")
    plt.xlabel("Layer"); plt.ylabel("Mean")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{mean_dir}/Vowel_{v}_lda_mean_across_layers.png", dpi=300, bbox_inches="tight")


for v in vowels_all:
    plt.figure(figsize=(7,5))
    for i, c in enumerate(components):
        plt.plot(layers, vowel_var_dicts[v][c], marker='o', label=c)
    plt.title(f"Variance LDA components across layers — vowel {v}")
    plt.xlabel("Layer"); plt.ylabel("Variance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{var_dir}/Vowel_{v}_lda_var_across_layers.png", dpi=300, bbox_inches="tight")


#######################################################################
# PCA the formants and the LDA components, do the same visualization as above
formant_df_with_pca_cols =  helpers.get_formant_with_pca_cols()

def plot_layer_distribution(component, num_layers):
    layer_stats = {}
    vowel_layer_stats = {}

    for layer_idx in range(1, num_layers):
        df = pd.read_pickle(f"./wav_2_vec_hidden_lda/lda_acts_{layer_idx}.pkl")
        
        X = df[["lda_1", "lda_2", "lda_3"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
        df[pca_cols] = X_pca

        components = [f"{component}_1", f"{component}_2", f"{component}_3"]
        missing = [c for c in components if c not in df.columns]
        if missing:
            print(f"Layer {layer_idx}: missing columns {missing}, skipping.")
            continue

        vowels = sorted(df["phone"].dropna().unique())

        outdir = f"./figures/pca_distributions/layer_{layer_idx}"
        os.makedirs(outdir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, comp in zip(axes, components):
            for v in vowels:
                subset = df.loc[df["phone"] == v, comp].dropna()
                ax.hist(subset, bins=30, alpha=0.5, label=v, density=True)
            ax.set_title(comp.upper())
            ax.set_xlabel("unit")
        axes[-1].legend(fontsize=8, ncol=2, title="vowel")

        fig.suptitle(f"{component} component distributions by vowel — Layer {layer_idx}")
        plt.tight_layout()
        fig.savefig(
            f"{outdir}/{component}_component_distributions_by_vowel.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        for v in vowels:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            for ax, comp in zip(axes, components):
                subset = df.loc[df["phone"] == v, comp].dropna()
                ax.hist(subset, bins=30, alpha=0.6, density=True)
                ax.set_title(comp.upper())
                ax.set_xlabel("unit")
                ax.set_ylabel("Count")

            fig.suptitle(f"{component} Component distributions for vowel {v}, layer {layer_idx}", fontsize=14)
            plt.tight_layout()

            out_path = f"{outdir}/vowel_{v}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        model = df[[f"{component}_1", f"{component}_2", f"{component}_3"]].to_numpy()
        model_mean = model.mean(axis=0)
        model_var = np.cov(model, rowvar=False)
        layer_stats[layer_idx] = (model_mean, model_var)

        vowel_layer_stats[layer_idx] = {}
        for v in vowels:
            vdf = df[df["phone"] == v][[f"{component}_1", f"{component}_2", f"{component}_3"]].to_numpy()
            v_mean = vdf.mean(axis=0)
            v_cov = np.cov(vdf, rowvar=False)
            vowel_layer_stats[layer_idx][v] = (v_mean, v_cov)

    return layer_stats, vowel_layer_stats


layer_stats, vowel_layer_stats = plot_layer_distribution(component="pca", num_layers=num_layers)


pca_1_mean_across_layers = {layer: layer_stats[layer][0][0] for layer in layer_stats}
pca_2_mean_across_layers = {layer: layer_stats[layer][0][1] for layer in layer_stats}
pca_3_mean_across_layers = {layer: layer_stats[layer][0][2] for layer in layer_stats}

pca_1_var_across_layers = {layer: layer_stats[layer][1][0][0] for layer in layer_stats}
pca_2_var_across_layers = {layer: layer_stats[layer][1][1][1] for layer in layer_stats}
pca_3_var_across_layers = {layer: layer_stats[layer][1][2][2] for layer in layer_stats}



# dicts you already created
mean_dicts = {
    "pca_1_mean": lda_1_mean_across_layers,
    "pca_2_mean": lda_2_mean_across_layers,
    "pca_3_mean": lda_3_mean_across_layers,
}

var_dicts = {
    "pca_1_var": lda_1_var_across_layers,
    "pca_2_var": lda_2_var_across_layers,
    "pca_3_var": lda_3_var_across_layers,
}

# ---- Plot means ----
plt.figure(figsize=(7,5))
for name, d in mean_dicts.items():
    layers = sorted(d.keys())
    values = [float(d[l]) for l in layers]
    plt.plot(layers, values, marker='o', label=name)

plt.title("Mean of PCA Components Across Layers")
plt.xlabel("Layer")
plt.ylabel("Mean Value")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
for name, d in var_dicts.items():
    layers = sorted(d.keys())
    values = [float(d[l]) for l in layers]
    plt.plot(layers, values, marker='o', label=name)

plt.title("Variance of PCA Components Across Layers")
plt.xlabel("Layer")
plt.ylabel("Variance")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()



summary_dir = "./figures/pca_distributions"
mean_dir = f"{summary_dir}/mean_across_layer"
var_dir = f"{summary_dir}/var_across_layer"

os.makedirs(mean_dir, exist_ok=True)
os.makedirs(var_dir, exist_ok=True)


vowels_all = sorted(formant_df["phone"].dropna().unique())
components = ["pca_1","pca_2","pca_3"]

# per vowel trajectories
vowel_mean_dicts = {v: {c: [] for c in components} for v in vowels_all}
vowel_var_dicts  = {v: {c: [] for c in components} for v in vowels_all}
layers = sorted(vowel_layer_stats.keys())

for layer in layers:
    for v in vowels_all:
        mean, cov = vowel_layer_stats[layer][v]
        for i, c in enumerate(components):
            vowel_mean_dicts[v][c].append(float(mean[i]))
            vowel_var_dicts[v][c].append(float(cov[i, i]))

for v in vowels_all:
    plt.figure(figsize=(7,5))
    for i, c in enumerate(components):
        plt.plot(layers, vowel_mean_dicts[v][c], marker='o', label=c)
    plt.title(f"Mean LDA components across layers — vowel {v}")
    plt.xlabel("Layer"); plt.ylabel("Mean")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{mean_dir}/Vowel_{v}_lda_mean_across_layers.png", dpi=300, bbox_inches="tight")


for v in vowels_all:
    plt.figure(figsize=(7,5))
    for i, c in enumerate(components):
        plt.plot(layers, vowel_var_dicts[v][c], marker='o', label=c)
    plt.title(f"Variance LDA components across layers — vowel {v}")
    plt.xlabel("Layer"); plt.ylabel("Variance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{var_dir}/Vowel_{v}_lda_var_across_layers.png", dpi=300, bbox_inches="tight")

#######################################################################
# calculate KLD between diff layers, for each vowel separately

across_layers_distr = {}
for layer, value in vowel_layer_stats.items():
    layer_distr = {}
    for vowel, (mean, cov) in value.items():
        layer_distr[vowel] = multivariate_normal(mean=mean, cov=cov)
    across_layers_distr[layer] = layer_distr

results = []

for l in range(1, num_layers):
    d1 = across_layers_distr.get(l, {})
    d2 = across_layers_distr.get(l + 1, {})

    vowels = sorted(set(d1.keys()) & set(d2.keys()))
    for v in vowels:
        try:
            kl = helpers.kld_calculate(d1[v], d2[v])  # KL(layer l || layer l+1) for vowel v
            results.append({
                "layer_from": l,
                "layer_to": l + 1,
                "vowel": v,
                "kl": float(kl),
            })
        except Exception as e:
            # keep going; record the error
            results.append({
                "layer_from": l,
                "layer_to": l + 1,
                "vowel": v,
                "kl": float("nan"),
                "error": str(e),
            })

kl_df = pd.DataFrame(results)
kl_df["layer_between"] = kl_df["layer_from"].astype(str) + "_" + kl_df["layer_to"].astype(str)


plt.figure(figsize=(8, 5))
for vowel, subdf in kl_df.groupby("vowel"):
    plt.plot(subdf["layer_between"], subdf["kl"], marker="o", label=vowel)

plt.xlabel("Layer pair (L→L+1)")
plt.ylabel("KL divergence (P‖Q)")
plt.title("KL divergence between consecutive layers (KL[lower layer | higher layer] per vowel")
plt.legend(title="Vowel", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
