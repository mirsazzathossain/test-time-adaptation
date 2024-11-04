import logging
import math
import os
from copy import deepcopy

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from methods.base import TTAMethod
from models.model import split_up_model
from utils.losses import Entropy, SymmetricCrossEntropy
from utils.misc import (
    compute_prototypes,
    confidence_condition,
    ema_update_model,
    ent_loss,
    init_pqs,
    plot_tsne,
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

        if cfg.TEST.WINDOW_LENGTH > 1:
            batch_size_src = cfg.TEST.BATCH_SIZE
        else:
            batch_size_src = cfg.TEST.WINDOW_LENGTH

        _, self.src_loader = get_source_loader(
            dataset_name=cfg.CORRUPTION.DATASET,
            adaptation=cfg.MODEL.ADAPTATION,
            preprocess=model.model_preprocess,
            data_root_dir=cfg.DATA_DIR,
            batch_size=batch_size_src,
            ckpt_path=cfg.MODEL.CKPT_PATH,
            num_samples=cfg.SOURCE.NUM_SAMPLES,
            percentage=cfg.SOURCE.PERCENTAGE,
            workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
            use_clip=cfg.MODEL.USE_CLIP,
        )
        self.src_loader_iter = iter(self.src_loader)
        self.contrast_mode = cfg.CONTRAST.MODE
        self.temperature = cfg.CONTRAST.TEMPERATURE
        self.base_temperature = self.temperature
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        self.lambda_ce_src = cfg.Ours.LAMBDA_CE_SRC
        self.lambda_ce_trg = cfg.Ours.LAMBDA_CE_TRG
        self.lambda_cont = cfg.Ours.LAMBDA_CONT
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM
        # arguments neeeded for warm up
        self.warmup_steps = cfg.Ours.NUM_SAMPLES_WARM_UP // batch_size_src
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

        # split up the T1 model
        self.backbone_t1, self.classifier_t1 = split_up_model(
            self.model_t1, arch_name, self.dataset_name
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
        self.backbone_t2, self.classifier_t2 = split_up_model(
            self.model_t2, arch_name, self.dataset_name
        )
        self.optimizer_backbone_t2 = self.setup_optimizer(
            self.backbone_t2.parameters(), 0.01
        )
        self.optimizer_classifier_t2 = self.setup_optimizer(
            self.classifier_t2.parameters(), 0.01
        )

        # setup student model
        self.model_s = self.copy_model(self.model)
        for param in self.model_s.parameters():
            param.detach_()

        # configure student model
        self.configure_model(self.model_s)
        self.params, _ = self.collect_params(self.model_s)
        lr = self.cfg.OPTIM.LR

        if len(self.params) > 0:
            self.optimizer_bn = self.setup_optimizer(self.params, lr)

        _ = self.get_number_trainable_params(self.params, self.model)

        # setup priority queues for prototype updates
        self.priority_queues = init_pqs(self.num_classes, max_size=10)

        # Why is it in init?
        if self.dataset_name == "cifar10_c":
            num_channels = 640
        elif self.dataset_name == "cifar100_c":
            num_channels = 1024

        self.projector = nn.Sequential(
            nn.Linear(num_channels, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        ).to(self.device)
        self.optimizer_t2.add_param_group(
            {
                "params": self.projector.parameters(),
                "lr": self.optimizer_t2.param_groups[0]["lr"],
            }
        )

    # new_s
    def contrastive_loss2(
        self, extracted_feature, prototypes, actual_labels, margin=0.5
    ):
        # Normalize the extracted features and prototypes
        extracted_feature = F.normalize(extracted_feature, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        # Compute cosine similarity between each extracted feature and all prototypes
        cosine_sim = torch.matmul(extracted_feature, prototypes.T)  # Shape: [200, 10]

        # Gather the positive similarities (correct prototype) for each sample
        pos_sim = cosine_sim[
            torch.arange(cosine_sim.size(0)), actual_labels
        ]  # Shape: [200]

        # Create a mask to ignore the correct class in negative similarities
        mask = torch.ones_like(cosine_sim, dtype=bool)
        mask[torch.arange(cosine_sim.size(0)), actual_labels] = False

        # Calculate the contrastive loss
        loss = 0.0
        for i in range(cosine_sim.size(0)):  # Iterate over batch samples
            # Get negative similarities for the current sample
            neg_sim = cosine_sim[i][mask[i]]  # Shape: [9]

            # Compute the margin-based contrastive loss
            losses = F.relu(margin - pos_sim[i] + neg_sim)
            loss += losses.mean()

        # Average the loss over the batch
        loss /= cosine_sim.size(0)

        return loss

    def contrastive_loss(
        self, features_test, prototypes_src, features_aug_test, labels=None, mask=None
    ):
        prototypes_src = prototypes_src.unsqueeze(1)  # 10, 1, 640
        with torch.no_grad():
            # dist[:, i] contains the distance from every source sample to one test sample
            x1 = prototypes_src.repeat(1, features_test.shape[0], 1)  # 10, 200, 640
            x2 = features_test.view(
                1, features_test.shape[0], features_test.shape[1]
            ).repeat(prototypes_src.shape[0], 1, 1)  # 10, 200, 640
            dist = F.cosine_similarity(x1, x2, dim=-1)  # 10, 200

            # for every test feature, get the nearest source prototype and derive the label
            _, indices = dist.topk(1, largest=True, dim=0)  # 1, 200
            indices = indices.squeeze(0)  # 200

        features = torch.cat(
            [
                prototypes_src[indices],
                features_test.view(features_test.shape[0], 1, features_test.shape[1]),
                features_aug_test.view(
                    features_test.shape[0], 1, features_test.shape[1]
                ),
            ],
            dim=1,
        )

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(
                self.device
            )  # 200, 200
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]  # 200
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 200*3, 640
        contrast_feature = self.projector(contrast_feature)  # 200*3, 128
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)  # 200*3, 128
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature  # 200*3, 128
            anchor_count = contrast_count  # 200
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )  # 200*3, 200*3
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 200*3, 1
        logits = anchor_dot_contrast - logits_max.detach()  # 200*3, 200*3

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # 200*3, 200*3

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )  # 200*3, 200*3

        mask = mask * logits_mask  # 200*3, 200*3

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 200*3, 200*3

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 200*3, 200*3

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 200*3

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos  # 200*3
        loss = loss.view(anchor_count, batch_size).mean()  # 1
        return loss

    def prototype_updates(
        self, priority_queues, num_classes, X_test_features, entropies, labels
    ):
        # Update priority queues with the current batch

        update_pqs(priority_queues, X_test_features, entropies, labels)

        # Print the current entropies in the priority queues after each batch
        if self.c % 5 == 0:
            popped_elements = pop_min_from_pqs(priority_queues, num_classes)
        # if self.c % 20 ==0:
        # print_queue_entropies(priority_queues, num_classes)

        # After all batches, compute the prototypes
        prototypes = compute_prototypes(
            priority_queues,
            num_classes,
            feature_dim=X_test_features.shape[1],
            device=X_test_features.device,
        )
        # print("Prototypes shape:", prototypes.shape)  # Should be [num_classes, feature_dim]

        # Plot TSNE for random 5 classes at interval of 100 steps
        if self.c % 20 == 0 and self.c > 0:
            # print("Plotting TSNE")
            plot_tsne(priority_queues, prototypes, num_classes, self.dataset_name)

        return prototypes

    def mse_feature_prototype(self, features, prototypes, labels):
        prototypes = prototypes[labels]  # [200,640]
        return F.mse_loss(features, prototypes, reduction="mean")

    def KL_Div_loss(self, features, prototypes, labels):
        prototypes = prototypes[labels]  # [200,640]
        # Apply softmax to get probability distributions
        prob1 = F.softmax(features, dim=1)
        prob2 = F.softmax(prototypes, dim=1)

        return F.kl_div(prob1.log(), prob2, reduction="batchmean")

    def loss_calculation(self, x):
        imgs_test = x[0]
        # print(imgs_test.shape)

        # forward augmented test data
        aug_hard_imgs_test = self.tta_transform(imgs_test)
        # aug_weak_imgs_test = self.tta_transform_2(imgs_test)

        outputs = self.model(imgs_test)
        outputs_stu = self.model_s(imgs_test)
        outputs_ema = self.model_t1(imgs_test)
        outputs_ema_2 = self.model_t2(imgs_test)

        outputs_stu_aug_hard = self.model_s(aug_hard_imgs_test)

        # loss = self.symmetric_cross_entropy(outputs_stu, outputs_ema.to(self.device)).mean(0) # + self.symmetric_cross_entropy(outputs_stu, outputs_ema_2.to(self.device)).mean(0)
        # loss1 = self.symmetric_cross_entropy(outputs_ema.to(self.device), outputs_ema_2.to(self.device)).mean(0)

        self.lambda_ce_trg = 1
        loss_self_training = (
            0.5 * self.symmetric_cross_entropy(outputs_stu, outputs_ema)
            + 0.5 * self.symmetric_cross_entropy(outputs_stu_aug_hard, outputs_ema)
            + 0.5 * self.symmetric_cross_entropy(outputs_stu, outputs_ema_2)
        ).mean(0)
        loss_stu = self.lambda_ce_trg * loss_self_training

        # all for prototype update
        entropy_stu = self.ent(outputs_stu)
        entropy_ema = self.ent(outputs_ema)
        entropy_ema_2 = self.ent(outputs_ema_2)

        filter_ids_1, filter_ids_2, filter_ids_3, filter_ids_4 = confidence_condition(
            entropy_ema, entropy_ema_2, entropy_threshold=0.4
        )

        # if self.c % 20==0:
        # print(filter_ids_2)
        # print(filter_ids_1[0].shape)
        # print(filter_ids_2[0].shape)
        # print(filter_ids_3[0].shape)
        # print(filter_ids_4[0].shape)
        # 1: T1, T2 both confident, #2: T1 conf, T2 not, #3: T1 not, T2 conf, #4: T1, T2 not
        # matching_ids, different_ids = get_matching_and_different_ids(outputs_ema, outputs_ema_2)

        self.backbone_t1, self.classifier_t1 = split_up_model(
            self.model_t1, self.arch_name, self.dataset_name
        )
        features_test = self.backbone_t1(imgs_test)
        # print("feature shape", features_test.shape)
        choice_filter_ids = filter_ids_2
        selected_features_candidates = features_test[
            choice_filter_ids
        ]  # selected candidates for prototype update
        selected_entropy_candidates = entropy_ema[choice_filter_ids]
        labels_ema = torch.argmax(outputs_ema, dim=1)
        selected_labels_candidates = labels_ema[choice_filter_ids]

        prototypes = self.prototype_updates(
            self.priority_queues,
            self.num_classes,
            selected_features_candidates,
            selected_entropy_candidates,
            selected_labels_candidates,
        )

        self.backbone_t2, self.classifier_t2 = split_up_model(
            self.model_t2, self.arch_name, self.dataset_name
        )  # could be memory exceed
        # labels_ema_2 = self.classifier_t2(prototypes)
        features_ema_2 = self.backbone_t2(imgs_test)
        features_aug_ema_2 = self.backbone_t2(aug_hard_imgs_test)

        # create and return the ensemble prediction
        stu_cof = 0
        ema_cof = 1
        ema_2_cof = 1
        outputs_ema = ema_cof * outputs_ema
        outputs_ema_2 = ema_2_cof * outputs_ema_2
        # outputs = stu_cof * outputs_stu + outputs_ema + outputs_ema_2
        outputs = torch.nn.functional.softmax(outputs_ema + outputs_ema_2, dim=1)

        loss1 = self.symmetric_cross_entropy(
            outputs_ema.to(self.device), outputs_ema_2.to(self.device)
        ).mean(0)

        loss2 = self.contrastive_loss2(
            features_ema_2, prototypes, labels_ema, margin=0.5
        )

        loss3 = self.mse_feature_prototype(features_ema_2, prototypes, labels_ema)

        loss4 = self.KL_Div_loss(features_ema_2, prototypes, labels_ema)

        loss5 = self.contrastive_loss(
            features_ema_2, prototypes, features_aug_ema_2, labels=None, mask=None
        )

        loss6 = ent_loss(
            outputs_ema_2
        )  # split to merge (information maximization loss) equation 14

        # if self.c % 20==0:
        # print("loss1 ", loss1.item())
        # print("loss2 " , loss2.item())
        # print("loss3 ", loss3.item())
        # print("loss4 ", loss4.item())
        # print("loss5 ", loss5.item())
        # print("loss6 ", loss6.item())

        l1_cof = 0
        l2_cof = 1
        l3_cof = 10
        l4_cof = 100
        l5_cof = 1
        l6_cof = 0
        loss_ema_2 = (
            l1_cof * loss1
            + l2_cof * loss2
            + l3_cof * loss3
            + l4_cof * loss4
            + l5_cof * loss5
            + l6_cof * loss6
        )

        return outputs, loss_stu, loss_ema_2, loss1

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, _, _ = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            with torch.amp.autocast("cuda"):
                outputs, loss, loss_ema_2, loss_classifier = self.loss_calculation(x)
                loss.requires_grad_(True)
                loss.backward(retain_graph=True)

                self.optimizer_bn.step()
                self.optimizer_bn.zero_grad()

                loss_ema_2.requires_grad_(True)
                loss_ema_2.backward()

                self.optimizer_backbone_t2.step()
                self.optimizer_backbone_t2.zero_grad()

        self.model_t1 = ema_update_model(
            model_to_update=self.model_t1,
            model_to_merge=self.model_s,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True,
        )

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
