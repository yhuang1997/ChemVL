import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from glob import glob
import json
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DCairo
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import math


def plot_loss_rocauc(loss_list, valid_metric_list, task_type, log_dir=None, regression_metric="rmse"):
    """
    :param regression_metric: for ``task_type == "regression"``, ``"rmse"`` (default) or ``"r2"`` etc.
        For ``"r2"``, values below ``-1`` are clipped to ``-1`` for plotting only; y-axis is fixed to
        ``[-1.05, 1.05]``. RMSE path plots raw values.
    """
    plt.figure(figsize=(10, 8))


    ax1 = plt.subplot(111)
    ax1.plot(loss_list, color="r", linestyle="dashed", label="loss")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()

    if task_type == "classification":
        ax2.plot(valid_metric_list, color="g", label="validation roc-auc")
        ax2.set_ylabel("roc-auc")
        ax2.set_ylim([0, 1])
    else:
        reg = (regression_metric or "rmse").lower()
        if reg == "r2":
            # R2 can be very negative early on; clip for display so the twin axis stays readable.
            vals = np.asarray(valid_metric_list, dtype=float)
            vals_plot = np.maximum(vals, -1.0)
            ax2.plot(vals_plot, color="g", label="validation R2 (<= -1 shown as -1)")
            ax2.set_ylabel("R2")
            ax2.set_ylim(-1.05, 1.05)
        else:
            ax2.plot(valid_metric_list, color="g", label="validation rmse")
            ax2.set_ylabel("rmse")

    plt.legend()
    plt.xlabel("* 5 epochs")
    plt.grid(True, linestyle="--")

    save_path = os.path.join(log_dir, "loss_roc-auc.png")
    plt.savefig(save_path)
    plt.close()


def plot_attention(attention, keys, axis, reduction='mean_batch', title_suffix='', topK=30, save_pickle=False,
                   save_dir=None):
    assert axis in ['bcpt', 'cpt', "bpt", "pt"], f"axis {axis} is not supported"
    assert reduction in ['mean_batch', 'mean'], f"reduction {reduction} is not supported"

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print("Directory {} is created.".format(save_dir))

    # bcpt -> cpt ;  bpt -> pt
    if "b" in axis:
        attention = attention.mean(axis=0)
    # *t -> *1 |-> cp1, p1
    if reduction == 'mean':
        attention = attention.mean(axis=-1, keepdims=True)
        # |-> cpt, pt
    for t in range(attention.shape[-1]):
        data = {key: att for key, att in zip(keys, attention[:, t])}

        if save_pickle and save_dir is not None:
            with open(os.path.join(save_dir, f"{title_suffix}_Task{t}.pkl"), "wb") as f:
                pickle.dump(data, f)

        sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True)[:topK])
        keys = list(sorted_data.keys())
        values = list(sorted_data.values())
        plt.figure(figsize=(8, 16))
        plt.barh(keys, values, color='blue')

        plt.title(f'{title_suffix}_T{t}_Top {topK}')
        plt.xlabel('Values')
        plt.ylabel('Keys')
        plt.gca().invert_yaxis()
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{title_suffix}_Task{t}_Top{topK}.png"))
        else:
            plt.show()
        plt.close()


def plot_ranked_attribute_values(folder, title_kwargs=None):
    pkl_files = glob(os.path.join(folder, "*.pkl"))
    T = len(pkl_files)
    data = {}
    for t in range(T):
        with open(pkl_files[t], "rb") as f:
            d = pickle.load(f)
            data.update({f"Epoch:{t}": d})
    num_attributes = len(d.mds())

    # Create a DataFrame to store data
    df = pd.DataFrame(data)

    # Calculate attribute rank for each time point
    ranked_data = df.apply(lambda col: rankdata(col, method='ordinal'), axis=0)
    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Color mapping
    colors = plt.cm.viridis(np.linspace(0, 1, num_attributes))

    for i in range(num_attributes):
        ax.plot(range(T), ranked_data.iloc[i, :], color=colors[i], alpha=0.75)

    # Create color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=num_attributes - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Attribute Index')

    ax.set_title('Ranked Attribute Values Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rank')
    ax.set_ylim(1, num_attributes)
    ax.invert_yaxis()  # Reverse y axis to place rank 1 at the top

    if title_kwargs is not None:
        title = ""
        for k, v in title_kwargs.items():
            title += f"{k}: {v}\n"
        ax.set_title(title)

    plt.show()


def plot_attention_v2(info, keys, axis, reduction='mean_batch', title_suffix='',
                   topK=30, task_id=0, save_dir=None):
    assert axis in ['bcpt', 'cpt', "bpt", "pt"], f"axis {axis} is not supported"
    assert reduction in ['mean_batch', 'mean'], f"reduction {reduction} is not supported"

    num_epochs = len(info["class_attention_info"])
    num_cols = 3  # Number of subplots per row
    if num_epochs % num_cols == 0:
        num_rows = num_epochs // num_cols
    else:
        num_rows = num_epochs // num_cols + 1
    # Create a large plot, set size
    plt.figure(figsize=(8 * num_cols, 4 * num_rows), dpi=100)

    info[f"class_attention_info_task{task_id}"] = {}
    for i in range(num_epochs):
        attention = info["class_attention_info"][i]

        # bcpt -> cpt ;  bpt -> pt
        if "b" in axis:
            attention = attention.mean(axis=0)

        # *t -> *1 |-> cp1, p1
        if reduction == 'mean':
            attention = attention.mean(axis=-1, keepdims=True)

        this_task_attention = attention[:, task_id] if axis == "bcpt" else attention[task_id]

        data = {key: att for key, att in zip(keys, this_task_attention)}
        info[f"class_attention_info_task{task_id}"][i] = data
        # Sort data by value and select top 30
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True)[:topK])
        # Extract sorted keys and values
        topK_keys = list(sorted_data.keys())
        values = list(sorted_data.values())

        # Determine current subplot position
        plt.subplot(num_rows, num_cols, i + 1)

        # Create horizontal bar chart
        plt.barh(topK_keys, values, color='blue')
        plt.title(f'Epoch_{i}_Top {topK}')
        plt.xlabel('Values')
        plt.ylabel('Keys')
        # Reverse y axis to place the bar with the highest value at the top
        plt.gca().invert_yaxis()
        # Adjust x axis label font size to prevent text overlap
        plt.xticks()
        plt.yticks()

    plt.tight_layout()

    # Display chart or save
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title_suffix}_Top{topK}.png"))
    else:
        plt.show()

    plt.close()


def plot_ranked_attribute_values_v2(pkl_file, task_id=0, title_kwargs=None, save_dir=None):
    data = pickle.load(open(pkl_file, "rb"))

    num_attributes = len(data["prior_keys"])
    num_epochs = len(data[f"class_attention_info_task{task_id}"])

    # Create a DataFrame to store data
    df = pd.DataFrame(data[f"class_attention_info_task{task_id}"])

    # Calculate attribute rank for each time point
    ranked_data = df.apply(lambda col: rankdata(col, method='ordinal'), axis=0)
    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Color mapping
    colors = plt.cm.viridis(np.linspace(0, 1, num_attributes))

    for i in range(num_attributes):
        ax.plot(range(num_epochs), ranked_data.iloc[i, :], color=colors[i], alpha=0.75)

    # Create color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=num_attributes - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Attribute Index')

    ax.set_title('Ranked Attribute Values Over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rank')
    ax.set_ylim(1, num_attributes)
    ax.invert_yaxis()  # Reverse y axis to place rank 1 at the top

    if title_kwargs is not None:
        title = ""
        for k, v in title_kwargs.items():
            title += f"{k}: {v}\n"
        ax.set_title(title)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "Ranked_Attribute_Values.png"))
    else:
        plt.show()


def plot_macro_metric(attention, keys, specific_epoch, title_suffix='', topk=30, save_dir=None, task_id=0, window_size=3):

    attention = np.array([attention[key] for key in attention.keys()])
    attention = np.take(attention, task_id, axis=-1)
    attention_in_window = attention[specific_epoch - window_size//2: specific_epoch + window_size//2 + 1]

    ws, num_batch, num_factor = attention_in_window.shape

    top_k_rates_list = []
    for w in range(ws):
        # Record top_k occurrence count for each factor in all samples
        factor_top_k_count = np.zeros(num_factor)
        this_attention = attention_in_window[w]
        # Iterate over each sample
        for i in range(num_batch):
            # Get current sample's top_k factor indices
            top_k_indices = np.argsort(this_attention[i])[-topk:]

            # For each top_k factor, increase count
            factor_top_k_count[top_k_indices] += 1

        # Calculate top_k rate for each factor
        top_k_rates = factor_top_k_count / num_batch
        top_k_rates_list.append(top_k_rates)
    top_k_rates_mean = np.mean(top_k_rates_list, axis=0)
    top_k_rates_std = np.std(top_k_rates_list, axis=0)
    data = {key: (mean, std) for key, mean, std in zip(keys, top_k_rates_mean, top_k_rates_std)}

    filtered_data = {key: (mean, std) for key, (mean, std) in data.items() if mean > 0.4}

    sorted_data = dict(sorted(filtered_data.items(), key=lambda item: item[1][0], reverse=True))

    sorted_means = [val[0] for val in sorted_data.values()]
    sorted_stds = [val[1] for val in sorted_data.values()]
    sorted_keys = list(sorted_data.keys())

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 6))
    plt.bar(sorted_keys, sorted_means, color='skyblue', edgecolor='black', align='center', yerr=sorted_stds, capsize=5)

    plt.title('Mean of Top-K Rates with Standard Deviation', fontsize=16)
    plt.ylabel('Mean Top-K Rate', fontsize=14)
    plt.xlabel('Chemical Descriptors', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add standard deviation annotation
    for i, (key, mean, std) in enumerate(zip(sorted_keys, sorted_means, sorted_stds)):
        plt.text(i, mean + 0.01, f'{mean:.2f}', ha='center', fontsize=12)

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title_suffix}_Top{topk}.png"))
    else:
        plt.show()



def highlight_descriptors(smiles, attention, filtered_keys=None, gt=None, topK=5,
                          save_dir=None, prefix='', show_values=True):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # filter out the unwanted descriptors
    attention = {k: v for k, v in attention.items() if k not in filtered_keys}

    # find the top K descriptors
    sorted_attention = sorted(attention.items(), key=lambda item: item[1], reverse=True)
    sorted_attention_topK = dict(sorted_attention[:topK])
    sorted_attention = dict(sorted_attention)

    total_hit_attention_score = sum(attention.values())
    average_hit_attention_score = total_hit_attention_score / len(attention)
    topK_hit_attention_score = sum(sorted_attention_topK.values())

    diff_ratio = {k: (v - average_hit_attention_score) / average_hit_attention_score * 100
                    for k, v in sorted_attention.items()}

    topK_diff_ratio = {k: (v - average_hit_attention_score) / average_hit_attention_score * 100
                          for k, v in sorted_attention_topK.items()}

    topK_attention = {k: v / topK_hit_attention_score for k, v in sorted_attention_topK.items()}

    drawer = MolDraw2DCairo(600, 600)
    drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.5], height_ratios=[1], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])

    topK_descriptors = list(topK_diff_ratio.keys())
    topK_weights = list(topK_diff_ratio.values())

    # barh show values
    ax1.barh(topK_descriptors, topK_weights,
             color='lightblue', edgecolor='black', label=f'Top {topK} Descriptors')

    if show_values:
        for i, (value, label) in enumerate(zip(topK_weights, topK_descriptors)):
            ax1.text(value + 0.5, i, f'{value:.2f}', va='center', fontsize=12, color='black')

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return diff_ratio


def highlight_descriptors_v2(smiles, attention, filtered_keys=None, gt=None, topK=5,
                             save_dir=None, prefix='', show_values=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    if filtered_keys is None:
        filtered_keys = []

    attention = {k: v for k, v in attention.items() if k not in filtered_keys}
    if not attention:
        raise ValueError("No descriptors remain after filtering; cannot highlight.")

    sorted_pairs = sorted(attention.items(), key=lambda item: item[1], reverse=True)
    sorted_attention = dict(sorted_pairs)
    sorted_attention_topK = dict(sorted_pairs[:topK])

    total_hit_attention_score = sum(attention.values())
    average_hit_attention_score = total_hit_attention_score / len(attention)

    diff_ratio = {k: (v - average_hit_attention_score) / average_hit_attention_score * 100
                    for k, v in sorted_attention.items()}

    topK_diff_ratio = {k: (v - average_hit_attention_score) / average_hit_attention_score * 100
                          for k, v in sorted_attention_topK.items()}

    drawer = MolDraw2DCairo(600, 600)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.5], height_ratios=[1], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    topK_descriptors = list(topK_diff_ratio.keys())
    topK_weights = list(topK_diff_ratio.values())
    ax1.barh(topK_descriptors, topK_weights,
             color='lightblue', edgecolor='black', label=f'Top {topK} Descriptors')
    ax1.set_xlabel('Relative Importance Compared with Average (%)', fontsize=12)

    if show_values:
        for i, (value, label) in enumerate(zip(topK_weights, topK_descriptors)):
            ax1.text(value + 0.5, i, f'{value:.2f}', va='center', fontsize=12, color='black')

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return diff_ratio
