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
        if len(pqs[class_label].queue) == pqs[class_label].max_size:
            min_entropy = pqs[class_label].pop_min()
            if min_entropy:
                min_entropies[class_label] = min_entropy
    return min_entropies

def plot_tsne(features, prototypes, true_labels):
    # Convert tensors to numpy arrays for t-SNE and visualization
    features_np = features.detach().cpu().numpy()
    prototypes_np = prototypes.detach().cpu().numpy()
    true_labels_np = true_labels.detach().cpu().numpy()

    # Concatenate features and prototypes for t-SNE
    all_features = np.vstack((features_np, prototypes_np))
    all_labels = np.concatenate((true_labels_np, np.arange(prototypes_np.shape[0])))  # Class labels for prototypes

    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    # Split the transformed data back into test features and prototypes
    test_tsne_results = tsne_results[:len(features_np)]
    prototype_tsne_results = tsne_results[len(features_np):]

    # Set up the plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(true_labels_np)
    palette = plt.get_cmap("tab10").colors

    # Plot test features with their ground truth labels
    for label, color in zip(unique_labels, palette):
        idx = true_labels_np == label
        plt.scatter(test_tsne_results[idx, 0], test_tsne_results[idx, 1],
                    color=color, label=f'Class {label}', alpha=0.6, edgecolor='k', s=40)

    # Plot prototypes with the same color as their corresponding class
    for label in range(prototypes_np.shape[0]):
        plt.scatter(prototype_tsne_results[label, 0], prototype_tsne_results[label, 1],
                    color=palette[label % len(palette)], marker='X', s=100, edgecolor='k')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")
    plt.title("t-SNE Visualization of Test Features and Class Prototypes")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
    plt.savefig(f"output/tsne_cifar100c_{wandb.run.id}.png")
    plt.savefig(f"output/tsne_cifar100c_{wandb.run.id}.pdf")
    wandb.log({"t-SNE": wandb.Image(f'output/tsne_cifar100c_{wandb.run.id}.png')})
    plt.show()
    plt.close()

def confidence_condition(entropy_ema, entropy_ema2, entropy_threshold):
    filter_ids = []
    filter_ids.append(torch.where((entropy_ema < entropy_threshold) & (entropy_ema2 < entropy_threshold)))
    filter_ids.append(torch.where((entropy_ema < entropy_threshold) & (entropy_ema2 > entropy_threshold)))
    filter_ids.append(torch.where((entropy_ema > entropy_threshold) & (entropy_ema2 < entropy_threshold)))
    filter_ids.append(torch.where((entropy_ema > entropy_threshold) & (entropy_ema2 > entropy_threshold)))

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


class DomainShiftScheduler(object):
    def __init__(self, optimizer, initial_lr=0.01, adjust_lr=None, decay_iterations=5):
        """
        Adjust the learning rate dynamically based on domain shift.

        Parameters:
        - optimizer: The optimizer you're using (e.g., SGD, Adam)
        - initial_lr: The original learning rate (default: 0.01)
        - adjust_lr: The learning rate to adjust to when domain shift is detected (None means no adjustment)
        - decay_iterations: The number of iterations to decay back to original learning rate (default: 5)
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.adjust_lr = adjust_lr
        self.decay_iterations = decay_iterations
        self.scheduler_counter = 0
        self.prev_im_loss = None

        if self.adjust_lr is not None:
            if self.adjust_lr == self.initial_lr:
                raise ValueError("adjust_lr should be different from initial_lr")
            if self.adjust_lr > self.initial_lr:
                self.adjust_type = "increase"
            else:
                self.adjust_type = "decrease"
        else:
            self.adjust_type = None

        if self.adjust_lr is not None:
            self.decay_factor = (self.initial_lr / self.adjust_lr) ** (
                1 / self.decay_iterations
            )
        else:
            self.decay_factor = 1

    def step(self, im_loss, threshold=0.1):
        if self.prev_im_loss is not None and im_loss - self.prev_im_loss > threshold:
            if self.adjust_lr is not None:
                if self.adjust_type == "increase":
                    logger.info(
                        f"Domain shift detected, increasing LR to {self.adjust_lr}"
                    )
                    self.optimizer.param_groups[0]["lr"] = self.adjust_lr
                elif self.adjust_type == "decrease":
                    logger.info(
                        f"Domain shift detected, decreasing LR to {self.adjust_lr}"
                    )
                    self.optimizer.param_groups[0]["lr"] = self.adjust_lr
                else:
                    logger.info("No adjustment to learning rate as adjust_lr is None.")
            else:
                logger.info("Domain shift detected, no adjustment in LR")

            self.scheduler_counter = self.decay_iterations

        if self.scheduler_counter > 0 and self.adjust_lr is not None:
            self.optimizer.param_groups[0]["lr"] *= (
                self.decay_factor
            )
            logger.info(f"Learing rate decayed to {self.optimizer.param_groups[0]['lr']}")
            self.scheduler_counter -= 1
            if self.scheduler_counter == 0:
                logger.info(f"Resetting learning rate to {self.initial_lr}")
                self.optimizer.param_groups[0]["lr"] = self.initial_lr

        self.prev_im_loss = im_loss
