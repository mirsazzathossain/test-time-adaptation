import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update


def print_memory_info():
    logger.info('-' * 40)
    mem_dict = {}
    for metric in ['memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved']:
        mem_dict[metric] = eval(f'torch.cuda.{metric}()')
        logger.info(f"{metric:>20s}: {mem_dict[metric] / 1e6:10.2f}MB")
    logger.info('-' * 40)
    return mem_dict

class PriorityQueue:
    """Priority queue to store the top-k prototypes with the highest entropy"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def add(self, feature, entropy) -> None:
        # If the queue is not full, add the new element
        # by maintaining the order of the queue
        if len(self.queue) < self.max_size:
            self.queue.append((entropy, feature))
            self.queue.sort(reverse=True, key=lambda x: x[0])
        else:
            # If the queue is full and the new entropy is lower than the 
            # max entropy (top element), replace the top element with 
            # the new one and re-sort the queue
            if entropy < self.queue[0][0]:
                self.queue[0] = (entropy, feature)
                self.queue.sort(reverse=True, key=lambda x: x[0])

    def get_sorted_features(self):
        return [item[1] for item in self.queue]

    def get_entropies(self):
        return [item[0] for item in self.queue]

    def pop_min(self):
        if self.queue:
            # Pop the last element (minimum entropy)
            return self.queue.pop() 
        return None
    
    def is_empty(self):
        return len(self.queue) == 0
    

def init_pqs(num_classes, max_size):
    pqs = defaultdict(lambda: PriorityQueue(max_size))
    return pqs

def update_pqs(pqs, features, entropies, labels):
    for i in range(features.size(0)):
        pqs[labels[i].item()].add(features[i], entropies[i])

def compute_prototypes(pqs, num_classes, feature_dim, device='cpu'):
    prototypes = []

    for class_label in range(num_classes):
        if pqs[class_label].queue:
            features = pqs[class_label].get_sorted_features()
            entropies = pqs[class_label].get_entropies()

            features = torch.stack([feature.to(device) for feature in features])
            entropies = torch.tensor(entropies).to(device)

            # Add small epsilon to avoid division by zero
            weights = 1/(entropies + 1e-6)

            # Reshape weights for broadcasting and apply to features
            weighted_features = features * weights.view(-1, 1)

            # Compute the prototype as the weighted sum of the features
            prototype = weighted_features.sum(dim=0) / weights.sum()
        else:
            prototype = torch.zeros(feature_dim).to(device)

        prototypes.append(prototype)

    return torch.stack(prototypes)

def log_queue(pqs, num_classes):
    for class_label in range(num_classes):
        entropies = pqs[class_label].get_entropies()
        if entropies:
            logger.info(f"Class {class_label}: {entropies}")
        else:
            logger.info(f"Class {class_label}: Empty")

def pop_min_from_pqs(pqs, num_classes):
    min_entropies = {}
    for class_label in range(num_classes):
        min_entropy = pqs[class_label].pop_min()
        if min_entropy:
            min_entropies[class_label] = min_entropy
    return min_entropies

def plot_tsne(pqs, prototypes, num_classes, dataset_name):
    tsne_features = []
    tsne_labels = []

    for class_label in range(num_classes):
        features = pqs[class_label].get_sorted_features()
        features = [feature.cpu().numpy() for feature in features]

        tsne_features.extend(features)
        tsne_labels.extend([class_label] * len(features))

        prototype = prototypes[class_label].cpu().numpy()
        tsne_features.append(prototype)
        tsne_labels.append(f'Prototype_{class_label}')

    tsne_features = np.array(tsne_features)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_features = tsne.fit_transform(tsne_features)

    plt.figure(figsize=(10, 8))
    for class_label in range(num_classes):
        indices = [i for i, label in enumerate(tsne_labels) if label == class_label]
        plt.scatter(tsne_features[indices, 0], tsne_features[indices, 1], label=f'Class {class_label}', alpha=0.6)

        prototype_index = tsne_labels.index(f'Prototype_{class_label}')
        plt.scatter(tsne_features[prototype_index, 0], tsne_features[prototype_index, 1], color='black', marker='x', s=100, label=f'Prototype {class_label}')


    plt.title(f't-SNE visualization of features and prototypes for each class in {dataset_name} dataset')
    plt.savefig(f"output/tsne_{dataset_name}_{wandb.run.id}.png")
    plt.show()

    wandb.log({"t-SNE": wandb.Image(f'output/tsne_{dataset_name}_{wandb.run.id}.png')})
    os.remove(f'output/tsne_{dataset_name}_{wandb.run.id}.png')
    plt.close()

def confidence_condition(entropy_ema, entropy_ema2, entropy_threshold):
    filter_ids = []
    filter_ids.append(torch.where((entropy_ema < entropy_threshold) & (entropy_ema2 < entropy_threshold)).item())
    filter_ids.append(torch.where((entropy_ema < entropy_threshold) & (entropy_ema2 > entropy_threshold)).item())
    filter_ids.append(torch.where((entropy_ema > entropy_threshold) & (entropy_ema2 < entropy_threshold)).item())
    filter_ids.append(torch.where((entropy_ema > entropy_threshold) & (entropy_ema2 > entropy_threshold)).item())

    return filter_ids

def get_matching_and_different_ids(out_1, out_2):
    pred_1 = torch.argmax(out_1, dim=1)
    pred_2 = torch.argmax(out_2, dim=1)

    matching_ids = torch.where(pred_1 == pred_2).nonzero(as_tuple=True)[0]
    different_ids = torch.where(pred_1 != pred_2).nonzero(as_tuple=True)[0]

    return matching_ids, different_ids

def print_queue_entropies(priority_queues, num_classes):
    for class_label in range(num_classes):
        entropies = priority_queues[class_label].get_entropies()
        print(f"Class {class_label}: Entropies = {entropies}")