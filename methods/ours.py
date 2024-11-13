import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from augmentations.transforms_cotta import get_tta_transforms
from methods.base import TTAMethod
from models.model import split_up_model
from utils.losses import (
    Entropy,
    L2SPLoss,
    MMDLoss,
    RMSNorm,
    SymmetricCrossEntropy,
    differential_loss,
    info_max_loss,
)
from utils.misc import (
    DomainShiftScheduler,
    compute_prototypes,
    confidence_condition,
    ema_update_model,
    init_pqs,
    pop_min_from_pqs,
    update_pqs,
)
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class Ours(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.c = 0
        self.contrast_mode = cfg.CONTRAST.MODE
        self.temperature = cfg.CONTRAST.TEMPERATURE
        self.base_temperature = self.temperature
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM
        self.final_lr = cfg.OPTIM.LR
        arch_name = cfg.MODEL.ARCH
        self.arch_name = arch_name

        # setup TTA transforms
        self.tta_transform = get_tta_transforms(self.img_size)

        # setup loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.ent = Entropy()

        # setup teacher model (T1)
        self.model_t1 = self.copy_model(self.model)
        for param in self.model_t1.parameters():
            param.detach_()

        # configure teacher model (T1)
        self.configure_model(self.model_t1, bn=True)
        self.params_t1, _ = self.collect_params(self.model_t1)
        lr = 0.01
        if len(self.params_t1) > 0:
            self.optimizer_t1 = self.setup_optimizer(self.params_t1, lr)

        _ = self.get_number_trainable_params(self.params_t1, self.model_t1)

        # split up the T1 model
        self.backbone_t1, _ = split_up_model(
            self.model_t1, self.arch_name, self.dataset_name
        )
        self.optimizer_backbone_t1 = self.setup_optimizer(
            self.backbone_t1.parameters(), 0.01
        )

        # setup teacher model (T2)
        self.model_t2 = self.copy_model(self.model)
        for param in self.model_t2.parameters():
            param.detach_()

        # configure teacher model (T2)
        self.configure_model(self.model_t2, bn=True)
        self.params_t2, _ = self.collect_params(self.model_t2)
        lr = 0.01
        if len(self.params_t2) > 0:
            self.optimizer_t2 = self.setup_optimizer(self.params_t2, lr)

        _ = self.get_number_trainable_params(self.params_t2, self.model_t2)

        # split up the T2 model and setup optimizers
        self.backbone_t2, _ = split_up_model(
            self.model_t2, self.arch_name, self.dataset_name
        )
        self.optimizer_backbone_t2 = self.setup_optimizer(
            self.backbone_t2.parameters(), 0.01
        )

        # setup student model
        self.model_s = self.copy_model(self.model)
        for param in self.model_s.parameters():
            param.detach_()

        # configure student model
        self.configure_model(self.model_s, bn=True)
        self.params_s, _ = self.collect_params(self.model_s)
        lr = self.cfg.OPTIM.LR

        if len(self.params_s) > 0:
            self.optimizer_s = self.setup_optimizer(self.params_s, lr)

        _ = self.get_number_trainable_params(self.params_s, self.model_s)

        # setup priority queues for prototype updates
        self.priority_queues = init_pqs(self.num_classes, max_size=10)

        # setup projector for contrastive loss
        if self.dataset_name == "cifar10_c":
            num_channels = 640
        elif self.dataset_name == "cifar100_c":
            num_channels = 1024
        else:
            num_channels = 2048

        self.projector = nn.Sequential(
            nn.Linear(num_channels, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        ).to(self.device)
        self.optimizer_backbone_t2.add_param_group(
            {
                "params": self.projector.parameters(),
                "lr": self.optimizer_t2.param_groups[0]["lr"],
            }
        )

        # split student model
        self.backbone_s, _ = split_up_model(
            self.model_s, self.arch_name, self.dataset_name
        )

        # keep a feature bank
        self.feature_bank = None

        # self.scheduler_s = DomainShiftScheduler(
        #     self.optimizer_s, self.optimizer_s.param_groups[0]["lr"], 0.1, 5
        # )
        # self.scheduler_backbone_t2 = DomainShiftScheduler(
        #     self.optimizer_backbone_t2,
        #     self.optimizer_backbone_t2.param_groups[0]["lr"],
        #     0.1,
        #     5,
        # )

    def prototype_updates(
        self, pqs, num_classes, features, entropies, labels, selected_feature_id
    ):
        """
        Update the priority queues and compute the prototypes for the current batch.

        Args:
            pqs (list): List of priority queues for each class
            num_classes (int): Number of classes
            features (Tensor): Extracted features for the current batch
            entropies (Tensor): Entropy values for the current batch
            labels (Tensor): Ground truth labels for the current batch

        Returns:
            Tensor: Prototypes for the current batch
        """
        # detach the features and entropies
        features = features.detach()
        entropies = entropies.detach()

        update_pqs(
            pqs,
            features[selected_feature_id],
            entropies[selected_feature_id],
            labels[selected_feature_id],
        )

        for class_label in range(num_classes):
            if not pqs[class_label].queue:
                # Get indices for the current class label
                class_indices = (labels == class_label).nonzero(as_tuple=True)[0]
                class_features = features[class_indices]
                class_entropies = entropies[class_indices]

                # Sort by entropy and select the top 5 minimum-entropy features for the class
                sorted_indices = torch.argsort(class_entropies)
                min_entropy_indices = (
                    sorted_indices[:5] if len(sorted_indices) >= 5 else sorted_indices
                )

                selected_features = class_features[min_entropy_indices]
                selected_entropies = class_entropies[min_entropy_indices]

                # Add selected features and entropies to the priority queue
                for feature, entropy in zip(selected_features, selected_entropies):
                    pqs[class_label].add(feature, entropy)

        # pop the minimum element from the priority queues every 5 batches
        if self.c % 50 == 0:
            _ = pop_min_from_pqs(pqs, num_classes)

        # compute the prototypes for the current batch
        prototypes = compute_prototypes(
            pqs,
            num_classes,
            feature_dim=features.shape[1],
            device="cuda",
        )

        return prototypes

    def is_pqs_full(self):
        """
        Check if the priority queues are full.

        Returns:
            bool: True if the priority queues are full, False otherwise
        """
        count = 0
        for i in range(self.num_classes):
            if self.priority_queues[i].is_empty():
                count += 1
        return count

    def loss_calculation(self, x, y=None):
        """
        Calculate the loss for the current batch.

        Args:
            x (Tensor): Input data for the current batch

        Returns:
            Tensor: Model predictions
            Tensor: Student model loss
            Tensor: T2 model loss
        """
        x = x[0]
        x_aug = self.tta_transform(x)

        outputs_s = self.model_s(x)
        outputs_t1 = self.model_t1(x)
        outputs_t2 = self.model_t2(x)
        outputs_stu_aug = self.model_s(x_aug)

        comb_t1_t2 = torch.nn.functional.softmax(outputs_t1 + outputs_t2, dim=1)
        comb_t1_t2_stu = torch.nn.functional.softmax(
            outputs_t1 + outputs_t2 + outputs_s, dim=1
        )
        comb_t1_s = torch.nn.functional.softmax(outputs_t1 + outputs_s, dim=1)
        comb_t2_s = torch.nn.functional.softmax(outputs_t2 + outputs_s, dim=1)

        logits_t1 = torch.nn.functional.softmax(outputs_t1, dim=1)
        logits_t2 = torch.nn.functional.softmax(outputs_t2, dim=1)
        logits_s = torch.nn.functional.softmax(outputs_s, dim=1)

        correct_t1 = torch.argmax(logits_t1, dim=1) == y
        correct_t2 = torch.argmax(logits_t2, dim=1) == y
        correct_s = torch.argmax(logits_s, dim=1) == y
        correct_comb_t1_t2 = torch.argmax(comb_t1_t2, dim=1) == y
        correct_comb_t1_t2_stu = torch.argmax(comb_t1_t2_stu, dim=1) == y
        correct_comb_t1_s = torch.argmax(comb_t1_s, dim=1) == y
        correct_comb_t2_s = torch.argmax(comb_t2_s, dim=1) == y

        total_correct_t1 = correct_t1.sum()
        total_correct_t2 = correct_t2.sum()
        total_correct_s = correct_s.sum()
        total_correct_comb_t1_t2 = correct_comb_t1_t2.sum()
        total_correct_comb_t1_t2_stu = correct_comb_t1_t2_stu.sum()
        total_correct_comb_t1_s = correct_comb_t1_s.sum()
        total_correct_comb_t2_s = correct_comb_t2_s.sum()

        error_t1 = 1 - total_correct_t1 / x.size(0)
        error_t2 = 1 - total_correct_t2 / x.size(0)
        error_s = 1 - total_correct_s / x.size(0)
        error_comb_t1_t2 = 1 - total_correct_comb_t1_t2 / x.size(0)
        error_comb_t1_t2_stu = 1 - total_correct_comb_t1_t2_stu / x.size(0)
        error_comb_t1_s = 1 - total_correct_comb_t1_s / x.size(0)
        error_comb_t2_s = 1 - total_correct_comb_t2_s / x.size(0)

        # actual accuracy of three models
        wandb.log(
            {
                "error_t1": error_t1,
                "error_t2": error_t2,
                "error_s": error_s,
                "error_comb_t1_t2": error_comb_t1_t2,
                "error_comb_t1_t2_stu": error_comb_t1_t2_stu,
                "error_comb_t1_s": error_comb_t1_s,
                "error_comb_t2_s": error_comb_t2_s,
            }
        )

        # final output
        outputs = torch.nn.functional.softmax(outputs_t1 + outputs_t2, dim=1)

        wandb.log(
            {"ce_t1_t2": self.symmetric_cross_entropy(outputs_t1, outputs_t2).mean(0)}
        )

        # student model loss
        loss_self_training = 0.0
        if "ce_s_t1" in self.cfg.Ours.LOSSES:
            loss_ce_s_t1 = self.symmetric_cross_entropy(outputs_s, outputs_t1.detach())
            loss_self_training += 0.5 * loss_ce_s_t1
            wandb.log({"ce_s_t1": loss_ce_s_t1.mean(0)})
        if "ce_s_t2" in self.cfg.Ours.LOSSES:
            loss_ce_s_t2 = self.symmetric_cross_entropy(outputs_s, outputs_t2.detach())
            loss_self_training += 0.5 * loss_ce_s_t2
            wandb.log({"ce_s_t2": loss_ce_s_t2.mean(0)})
        if "ce_s_aug_t1" in self.cfg.Ours.LOSSES:
            loss_ce_s_aug_t1 = self.symmetric_cross_entropy(
                outputs_stu_aug, outputs_t1.detach()
            )
            loss_self_training += 0.5 * loss_ce_s_aug_t1
            wandb.log({"ce_s_aug_t1": loss_ce_s_aug_t1.mean(0)})
        loss_stu = loss_self_training.mean(0)
        wandb.log({"loss_stu_ce": loss_stu})

        # calculate the entropy of the outputs
        entropy_t1 = self.ent(outputs_t1)
        entropy_t2 = self.ent(outputs_t2)

        # apply filtering for feature selection
        filter_ids_1, filter_ids_2, filter_ids_3, filter_ids_4 = confidence_condition(
            entropy_t1, entropy_t2, entropy_threshold=0.5
        )
        selected_filter_ids = filter_ids_2

        # select prototypes from T1 model
        features_t1 = self.backbone_t1(x)
        labels_t1 = torch.argmax(outputs_t1, dim=1)

        prototypes = self.prototype_updates(
            self.priority_queues,
            self.num_classes,
            features_t1,
            entropy_t1,
            labels_t1,
            selected_filter_ids,
        )

        if self.c % 200 == 0:
            logger.info(f"Number of empty queues: {self.is_pqs_full()}")

        # calculate the loss for the T2 model
        features_t2 = self.backbone_t2(x)
        features_aug_t2 = self.backbone_t2(x_aug)

        cntrs_t2_proto = self.contrastive_loss_proto(
            features_t2, prototypes.detach(), labels_t1, margin=0.5
        )
        mse_t2 = F.mse_loss(
            features_t2, prototypes[labels_t1].detach(), reduction="mean"
        )
        kld_t2 = self.KL_Div_loss(features_t2, prototypes.detach(), labels_t1)
        cntrs_t2 = self.contrastive_loss(
            features_t2, prototypes.detach(), features_aug_t2, labels=None, mask=None
        )
        im_loss = info_max_loss(outputs)

        loss_t2 = 0.0
        if "contr_t2_proto" in self.cfg.Ours.LOSSES:
            # loss_t2 += cntrs_t2_proto
            wandb.log({"contr_t2_proto": cntrs_t2_proto})
        if "mse_t2_proto" in self.cfg.Ours.LOSSES:
            # loss_t2 += 10 * mse_t2
            wandb.log({"mse_t2_proto": 10 * mse_t2})
        if "kld_t2_proto" in self.cfg.Ours.LOSSES:
            # loss_t2 += 100 * kld_t2
            wandb.log({"kld_t2_proto": 100 * kld_t2})
        if "contr_t2" in self.cfg.Ours.LOSSES:
            loss_t2 += cntrs_t2
            wandb.log({"contr_t2": cntrs_t2})
        if "im_loss" in self.cfg.Ours.LOSSES:
            loss_t2 += 2 * im_loss
            wandb.log({"im_loss": im_loss})

        features_s = self.backbone_s(x)
        if self.c == 0:
            mem_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            self.ghajini = MMDLoss()
            mem_loss = self.ghajini(self.feature_bank.detach(), features_s)

        self.feature_bank = features_s

        if "l2_sp" in self.cfg.Ours.LOSSES:
            pretrained_weights = self.model_states[0]
            loss_l2_sp = L2SPLoss(pretrained_weights)
            l2_sp = loss_l2_sp(self.model_s)
            loss_stu += l2_sp
            wandb.log({"l2_sp": l2_sp})

        if "mem_loss" in self.cfg.Ours.LOSSES:
            loss_stu += mem_loss
            wandb.log({"mem_loss": mem_loss})

        # self.scheduler_s.step(im_loss, threshold=0.8)
        # self.scheduler_backbone_t2.step(im_loss, threshold=0.8)

        wandb.log({"loss_stu": loss_stu})
        wandb.log({"loss_t2": loss_t2})

        return outputs, loss_stu, loss_t2

    @torch.enable_grad()
    def forward_and_adapt(self, x, y=None):
        """
        Forward pass and adaptation for the current batch.

        Args:
            x (Tensor): Input data for the current batch

        Returns:
            Tensor: Model predictions
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, _ = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            with torch.amp.autocast("cuda"):
                outputs, loss_stu, loss_t2 = self.loss_calculation(x, y)

                self.optimizer_s.zero_grad()
                loss_stu.backward()
                self.optimizer_s.step()

                self.optimizer_backbone_t2.zero_grad()
                loss_t2.backward()
                self.optimizer_backbone_t2.step()

        self.model_t1 = ema_update_model(
            model_to_update=self.model_t1,
            model_to_merge=self.model_s,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True,
        )

        self.model_t2 = ema_update_model(
            model_to_update=self.model_t2,
            model_to_merge=self.model_s,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True,
        )

        # Stochastic restore
        with torch.no_grad():
            self.rst = 0.01
            if self.rst > 0.0:
                for nm, m in self.model_t2.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ["weight", "bias"] and p.requires_grad:
                            mask = (
                                (torch.rand(p.shape) < self.rst).float().to(self.device)
                            )
                            p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (
                                1.0 - mask
                            )

        with torch.no_grad():
            if True:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(
                    prior
                )
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

        self.c = self.c + 1
        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        outputs_test = self.model(imgs_test)
        outputs_ema = self.model_t1(imgs_test)
        return outputs_test + outputs_ema

    def configure_model(self, model=None, bn=None):
        """
        Configure model

        Options:
        - configure_model() : as same as original
        - configure_model(model) : configure model custom
        - configure_model(model, bn=True) : configure model with bn
        - configure_model(model, bn=False) : configure model without bn
        """
        model = model if model is not None else self.model
        model.eval()
        model.requires_grad_(False)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn is None or bn:
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                if bn is None or bn:
                    m.requires_grad_(True)
            else:
                m.requires_grad_(False if bn else True)

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_states = [
            deepcopy(optimizer.state_dict()) for optimizer in self.optimizers
        ]
        return model_states, optimizer_states

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        for optimizer, optimizer_state in zip(self.optimizers, self.optimizer_states):
            optimizer.load_state_dict(optimizer_state)

    def KL_Div_loss(self, features, prototypes, labels):
        """
        Compute the KL divergence loss between the features and prototypes.

        Args:
            features (Tensor): Extracted features for the current batch
            prototypes (Tensor): Prototypes for the current batch
            labels (Tensor): Ground truth labels for the current batch

        Returns:
            Tensor: KL divergence loss
        """
        prototypes = prototypes[labels]
        prob1 = F.softmax(features, dim=1)
        prob2 = F.softmax(prototypes, dim=1)

        return F.kl_div(prob1.log(), prob2, reduction="batchmean")

    def contrastive_loss_proto(self, feature, prototypes, labels, margin=0.5):
        """
        Compute the contrastive loss between the features and prototypes.

        Args:
            feature (Tensor): Extracted features for the current batch
            prototypes (Tensor): Prototypes for the current batch
            labels (Tensor): Ground truth labels for the current batch
            margin (float): Margin value for the contrastive loss

        Returns:
            Tensor: Contrastive loss
        """
        # normalize the features and prototypes
        feature = F.normalize(feature, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        # compute the cosine similarity between features and prototypes
        cosine_sim = torch.matmul(feature, prototypes.T)

        # get the positive similarities (correct class)
        pos_sim = cosine_sim[torch.arange(cosine_sim.size(0)), labels]

        # mask to ignore the correct class in negative similarities
        mask = torch.ones_like(cosine_sim, dtype=bool)
        mask[torch.arange(cosine_sim.size(0)), labels] = False

        # compute the loss
        loss = 0.0
        for i in range(cosine_sim.size(0)):
            neg_sim = cosine_sim[i][mask[i]]
            losses = F.relu(margin - pos_sim[i] + neg_sim)
            loss += losses.mean()

        loss /= cosine_sim.size(0)

        return loss

    def contrastive_loss(
        self, features, prototypes, features_aug, labels=None, mask=None
    ):
        """
        Compute the contrastive loss.

        Args:
            features (Tensor): Extracted features for the current batch
            prototypes (Tensor): Prototypes for the current batch
            features_aug (Tensor): Augmented features for the current batch
            labels (Tensor): Ground truth labels for the current batch
            mask (Tensor): Mask for the contrastive loss

        Returns:
            Tensor: Contrastive loss
        """
        prototypes = prototypes.unsqueeze(1)
        with torch.no_grad():
            x1 = prototypes.repeat(1, features.shape[0], 1)
            x2 = features.view(1, features.shape[0], features.shape[1]).repeat(
                prototypes.shape[0], 1, 1
            )
            dist = F.cosine_similarity(x1, x2, dim=-1)

            # get the indices of the nearest prototypes
            _, indices = dist.topk(1, largest=True, dim=0)
            indices = indices.squeeze(0)

        features = torch.cat(
            [
                prototypes[indices],
                features.view(features.shape[0], 1, features.shape[1]),
                features_aug.view(features.shape[0], 1, features.shape[1]),
            ],
            dim=1,
        )

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
