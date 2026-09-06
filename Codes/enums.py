# -*- coding: utf-8 -*-
# Codes/enums.py

from Codes.datasetHelper import DatasetType
from Models.DNNs import resnet, lenet5


class Config:
    moduleName            = None
    DB_samples_per_client = None
    DB_nonIID_alpha       = None
    num_clients           = None      # number of federated clients
    rounds                = None      # communication rounds
    local_epochs          = None      # local epochs per client
    distill_epochs        = None      # server-side distillation epochs
    configString          = None
    currentEpoch          = None
    model                 = None
    datasetSel            = None
    trainStrategy         = None
    studentsStrategy      = None
    local_batch_size      = None
    eval_batch_size       = None
    mask_gradient_batch_size = None
    tf32                  = None
    distill_batch_size    = None
    temperature           = None      # distillation temperature
    seed                  = None
    verbose               = None
    DB_nonIID             = None
    DB_globalDatasetEN    = None
    DB_forceCreate        = None
    DB_dataset            = None
    createDB              = None
    dataSetAddress        = None
    excelAddr             = None
    saveAddr              = None
    saveAddr_iter         = None
    currentRound          = None
    classNum              = None
    inputShape            = None
    currentEdge           = None
    gpu_id                = None
    encryption_ratio      = None
    aggregate_BN          = None
    attack                = None        # run Membership Inference Attack & Gradient Inversion Attacks
    attack_interval       = None
    run_mia               = None
    run_dlg               = None
    attack_seed           = None
    mia_sample_size       = None
    mia_bootstrap_samples = None
    dlg_num_samples       = None
    dlg_num_restarts      = None
    dlg_iterations        = None
    dlg_learning_rate     = None
    dlg_optimizer         = None
    dlg_objective         = None
    dlg_tv_weight         = None
    dlg_early_stopping_patience = None
    dlg_success_ssim      = None
    dlg_compute_lpips     = None
    dlg_known_label       = None
    run_ig                = None
    ig_num_samples        = None
    ig_num_restarts       = None
    ig_iterations         = None
    ig_learning_rate      = None
    ig_tv_weight          = None
    ig_early_stopping_patience = None
    ig_success_ssim       = None
    ig_compute_lpips      = None
    ig_known_label        = None
    run_ilrg              = None
    ilrg_batch_size       = None
    ilrg_num_batches      = None
    ilrg_alpha            = None
    ilrg_mask_mode        = None
    checkpoint_root       = None
    resolved_args         = None
    # Experiment metadata
    exp_id = 'default'
    run_id = None
    group = 'default'

    # Differential Privacy
    # dp_epsilon = 1.0
    # dp_delta = 1e-5

    # Privacy evaluation settings
    # privacy_eval_frequency = 10  # Evaluate every N rounds
    # privacy_attack_iterations = 1500
    # privacy_attack_lr = 0.1
    # save_privacy_results = True


def resolve_dataset(name: str):
    name = name.upper()
    if name == "CIFAR10":
        return DatasetType.CIFAR10
    if name == "MNIST":
        return DatasetType.MNIST
    if name == "FMNIST" :
        return DatasetType.FMNIST
    if name == "SVHN":
        return DatasetType.SVHN
    raise ValueError(f"Unknown dataset: {name}")


def resolve_model(name: str):
    """
    Resolves and returns the uninstantiated blueprint class reference.
    This allows TrainEngine to instantiate the network internally with correct setup shapes.
    """
    name = name.upper()
    if name == "LENET5":
        return lenet5.Lenet5

    if name == "RESNET18":
        return resnet.ResNet18

    if name == "RESNET34":
        return resnet.ResNet34

    raise ValueError(f"Unknown model: {name}")
