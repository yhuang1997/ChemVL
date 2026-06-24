import torch
import numpy as np
import torch.nn.functional as F


@torch.no_grad()
def calculate_attention_on_multitask(task_type, model, data_loader, device):
    model.eval()

    attention_mode = model.prior_fusion_block.mode
    attention_list = []
    label_list = []
    sample_num = 0
    for step, data in enumerate(data_loader):
        if len(data) == 2:
            images, labels = data
            smiles = None
        else:
            images, labels, smiles = data
        images, labels = images.to(device), labels.to(device)
        label_list.append(labels)
        sample_num += images.shape[0]

        with torch.no_grad():
            # predict tasks separately, or causes Memory Overflow!!
            att = []
            for task_id in range(labels.shape[1]):
                # [batch_size, num_classes, num_priors, 1]
                this_att = model.calculate_text_knowledge_attention(smiles=smiles, selected_task_ids=[task_id])
                att.append(this_att)

            att = torch.cat(att, dim=-1)
        attention_list.append(att)
        # static_attention is agnostic to the number of samples
        if attention_mode == "static_attention":
            break

    labels = torch.cat(label_list, dim=0).cpu().numpy()  # (sample_num, num_tasks)
    if task_type == "regression":
        discreted_labels = np.zeros_like(labels, dtype=int)
        for t in range(labels.shape[1]):
            for i in range(labels.shape[0]):
                discreted_labels[i, t] = model.rds.get_bin_index(t, labels[i, t])
        labels = discreted_labels

    attention = torch.cat(attention_list, dim=0).cpu().numpy()
    if len(attention.shape) == 4:
        sample_num, num_classes, num_priors, num_tasks = attention.shape
        class_attention = np.zeros((sample_num, num_priors, num_tasks))
        for i in range(sample_num):
            for j in range(num_tasks):
                class_idx = labels[i, j]
                class_attention[i, :, j] = attention[i, class_idx, :, j]
        axis = "bcpt"  # batch, class, prior, task
    elif len(attention.shape) == 3:
        num_classes, num_priors, num_tasks = attention.shape
        class_attention = np.zeros((num_priors, num_tasks))
        for j in range(num_tasks):
            class_idx = labels[j]
            class_attention[:, j] = attention[class_idx, :, j]
        axis = "cpt"
    else:
        raise ValueError(f"attention shape {attention.shape} is not supported")

    return attention, class_attention, axis
