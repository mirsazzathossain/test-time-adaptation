import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import wandb

import methods
from conf import cfg, ckpt_path_to_domain_seq, get_num_classes, load_cfg_from_args
from datasets.data_loading import get_test_loader
from models.model import get_model
from utils.eval_utils import eval_domain_dict, get_accuracy
from utils.misc import print_memory_info
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated",
                      "continual_mixed_domain",
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    # setup wandb logging
    wandb.run.name = cfg.MODEL.ADAPTATION + "-" + cfg.SETTING + "-" + cfg.CORRUPTION.DATASET

    information = "remove_loss_ce_s_t2"
    wandb.run.name += "-" + information

    # add current bangladesh time to the run name
    now = datetime.now()
    new_time = now + timedelta(hours=11)
    wandb.run.name += "-" + new_time.strftime("%Y-%m-%d-%H-%M-%S")

    wandb.config.update(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # ==========================================================================
    if cfg.MODEL.ADAPTATION == "ours":
        loss_name = {
            "ce_s_t1": "Symmetric Cross Entropy (T1)",
            "ce_s_t2": "Symmetric Cross Entropy (T2)",
            "ce_s_aug_t1": "Symmetric Cross Entropy (Aug X to T1)",
            "contr_t2_proto": "Contrastive (T2 - Prototype)",
            "mse_t2_proto": "MSE (T2 - Prototype)",
            "contr_t2": "Contrastive (T2 - Aug and Prototype)",
            "im_loss": "Information Maximization Loss (T2)",
            "differ_loss": "Differential Loss (S - T1 - T2)",
            "mem_loss": "Maximum Mean Discrepancy Loss (T2)",
            "kld_t2_proto": "KL Divergence Loss (T2 - Prototype)",
        }

        desc = f"\n\nDataset: {cfg.CORRUPTION.DATASET}\nSetup: {cfg.SETTING}\nTraining Strategy: Batch Normalization (T1, T2), All Layers (S)\nLoss: \n"

        student_losses = [
            loss_name[loss]
            for loss in cfg.Ours.LOSSES
            if loss in ["ce_s_t1", "ce_s_t2", "ce_s_aug_t1", "differ_loss"]
        ]
        t2_losses = [
            loss_name[loss]
            for loss in cfg.Ours.LOSSES
            if loss
            in [
                "contr_t2_proto",
                "mse_t2_proto",
                "contr_t2",
                "im_loss",
                "mem_loss",
                "kld_t2_proto",
            ]
        ]

        desc += "  - S: " + ", ".join(student_losses) + "\n"
        desc += "  - T1: EMA using S weights\n"
        desc += "  - T2: " + ", ".join(t2_losses) + "\n"

        wandb.summary["description"] = desc
        logger.info(desc)
    # ==========================================================================

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using {cfg.CORRUPTION.DATASET} with the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}

    # start evaluation
    for i_dom, domain_name in enumerate(domain_seq_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(
                setting=cfg.SETTING,
                adaptation=cfg.MODEL.ADAPTATION,
                dataset_name=cfg.CORRUPTION.DATASET,
                preprocess=model_preprocess,
                data_root_dir=cfg.DATA_DIR,
                domain_name=domain_name,
                domain_names_all=domain_sequence,
                severity=severity,
                num_examples=cfg.CORRUPTION.NUM_EX,
                rng_seed=cfg.RNG_SEED,
                use_clip=cfg.MODEL.USE_CLIP,
                n_views=cfg.TEST.N_AUGMENTATIONS,
                delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
            )

            if i_dom == 0:
                # Note that the input normalization is done inside of the model
                logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")

            # evaluate the model
            acc, domain_dict, num_samples = get_accuracy(
                model,
                data_loader=test_data_loader,
                dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name,
                setting=cfg.SETTING,
                domain_dict=domain_dict,
                print_every=cfg.PRINT_EVERY,
                device=device
            )

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")

    if cfg.SETTING == "continual_mixed_domain":
        i_dom, domain_name = 0, 'mixed'

        for severity in severities:
            test_data_loader = get_test_loader(
                setting='mixed_domains',
                adaptation=cfg.MODEL.ADAPTATION,
                dataset_name=cfg.CORRUPTION.DATASET,
                preprocess=model_preprocess,
                data_root_dir=cfg.DATA_DIR,
                domain_name=domain_name,
                domain_names_all=domain_sequence,
                severity=severity,
                num_examples=cfg.CORRUPTION.NUM_EX,
                rng_seed=cfg.RNG_SEED,
                use_clip=cfg.MODEL.USE_CLIP,
                n_views=cfg.TEST.N_AUGMENTATIONS,
                delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
            )

            if i_dom == 0:
                # Note that the input normalization is done inside of the model
                logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")

            # evaluate the model
            acc, domain_dict, num_samples = get_accuracy(
                model,
                data_loader=test_data_loader,
                dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name,
                setting='mixed_domains',
                domain_dict=domain_dict,
                print_every=cfg.PRINT_EVERY,
                device=device
            )

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)

    if cfg.TEST.DEBUG:
        print_memory_info()

    # save the ckpt to wandb
    wandb.save(cfg.CKPT_DIR)
    wandb.finish()

if __name__ == '__main__':
    
    # ==========================================================================
    wandb_api_key=os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("WANDB_API_KEY not found in environment variables.")
    # ========================================================================== 
    
    wandb.init(project="tta", dir="output")

    evaluate('"Evaluation.')
