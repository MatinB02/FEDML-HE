# -*- coding: utf-8 -*-
# ProjectControl_Loop.py  –  migrated from TensorFlow 1.x to PyTorch

# ############################################ imports ############################################
import os
import json
import random
gpu_id = 0  # choose the physical GPU you want to use (0 or 1 on server4)

import torch
import numpy as np
import gc
import argparse
import time

from pathlib import Path
from Codes.functions_mainAlg import build_mask_plan, flattener, extractStructure, reconstructor, indexMask_to_BinaryMask
from Codes import excelHelper
from Codes.CKKSRun import HE_CKKS
from Codes.datasetHelper import DatasetLoader
from Codes.functions import delete_all_in_folder, foldersInit, maskFilter
from Codes.functions_mainAlg import FilteredDNNEncryption
from Codes.path_utils import safe_path_component
from Codes.trainEngine import TrainEngine
from Codes.enums import Config, resolve_dataset, resolve_model
from Codes.attack_checkpoints import AttackCheckpointWriter, generate_run_id


# ############################################ parse_args function ############################################
def parse_args():
    p = argparse.ArgumentParser("FEDML-HE experiments")

    # Federated learning
    p.add_argument("--rounds", type=int, default=2)             # communication rounds
    p.add_argument("--num_clients", type=int, default=2)        # number of federated clients
    p.add_argument("--local_epochs", type=int, default=5)       # local epochs per client
    p.add_argument("--encryption_ratio", type=float, default=0.03)

    # we always send encrypted BN to server, and the server aggregates them.
    # if aggregate_BN: at the beginning of each round, clients use the aggregated BNs.
    # else: each client uses it's local BNs from previous round.
    p.add_argument("--aggregate_BN", action="store_true", default=False)

    # Dataset
    p.add_argument("--dataset", type=str, default="CIFAR10")
    p.add_argument("--DB_samples_per_client", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nonIID", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--DB_forceCreate", action="store_true", default=False)

    # Model
    p.add_argument("--model", type=str, default="lenet5")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--local_batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--mask_gradient_batch_size", type=int, default=128)
    p.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=True,
        help="Use TensorFloat-32 for CUDA matrix multiplications and convolutions.",
    )

    # Misc
    p.add_argument("--group", type=str, default="MANUAL")
    p.add_argument("--exp_id", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--checkpoint_root", type=str, default="./Results/AttackCheckpoints")

    # Attacks
    p.add_argument("--attack", action=argparse.BooleanOptionalAction, default=True)     # Capture offline-attack checkpoints
    p.add_argument("--attack_interval", type=int, default=2)                            # capture once per x rounds
    p.add_argument("--run_mia", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_dlg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attack_seed", type=int, default=2026)
    p.add_argument("--mia_sample_size", type=int, default=500)
    p.add_argument("--mia_bootstrap_samples", type=int, default=1000)
    p.add_argument("--run_ilrg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ilrg_batch_size", type=int, default=None)
    p.add_argument("--ilrg_num_batches", type=int, default=3)
    p.add_argument("--ilrg_alpha", type=float, default=0.01)
    p.add_argument("--ilrg_mask_mode", choices=("partial", "strict"), default="partial")
    p.add_argument("--dlg_num_samples", type=int, default=3)
    p.add_argument("--dlg_num_restarts", type=int, default=3)
    p.add_argument("--dlg_iterations", type=int, default=1000)
    p.add_argument("--dlg_learning_rate", type=float, default=0.01)
    p.add_argument("--dlg_optimizer", choices=("adam", "lbfgs"), default="adam")
    p.add_argument("--dlg_objective", choices=("l2", "normalized_l2", "cosine"), default="l2")
    p.add_argument("--dlg_tv_weight", type=float, default=1e-4)
    p.add_argument("--dlg_early_stopping_patience", type=int, default=200)
    p.add_argument("--dlg_success_ssim", type=float, default=0.5)
    p.add_argument("--dlg_compute_lpips", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dlg_known_label", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--run_ig", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ig_num_samples", type=int, default=3)
    p.add_argument("--ig_num_restarts", type=int, default=1)
    p.add_argument("--ig_iterations", type=int, default=4800)
    p.add_argument("--ig_learning_rate", type=float, default=0.1)
    p.add_argument("--ig_tv_weight", type=float, default=1e-4)
    p.add_argument("--ig_early_stopping_patience", type=int, default=200)
    p.add_argument("--ig_success_ssim", type=float, default=0.5)
    p.add_argument("--ig_compute_lpips", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ig_known_label", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()

# ############################################ set config parameters ############################################
def build_config():
    cfg = Config()
    args = parse_args()
    cfg.moduleName = "FEDML-HE"
    cfg.dataSetAddress = str(Path(__file__).absolute()).split('FEDML-HE_torch')[0] + 'Datasets'
    cfg.gpu_id = gpu_id
    cfg.num_clients = args.num_clients
    cfg.local_epochs = args.local_epochs
    cfg.rounds = args.rounds
    cfg.encryption_ratio = args.encryption_ratio
    cfg.aggregate_BN = args.aggregate_BN
    cfg.seed = args.seed
    cfg.DB_nonIID_alpha = args.alpha        # we are currently assuming non-IID all the time
    cfg.temperature = args.temperature
    cfg.DB_nonIID = args.nonIID
    cfg.DB_forceCreate = args.DB_forceCreate
    cfg.DB_samples_per_client = args.DB_samples_per_client
    cfg.DB_dataset = resolve_dataset(args.dataset)
    cfg.model = resolve_model(args.model)
    cfg.local_batch_size = args.local_batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.mask_gradient_batch_size = args.mask_gradient_batch_size
    cfg.tf32 = args.tf32
    cfg.group = args.group
    cfg.exp_id = args.exp_id
    cfg.checkpoint_root = args.checkpoint_root
    cfg.attack = args.attack
    cfg.attack_interval = args.attack_interval
    cfg.run_mia = args.run_mia
    cfg.run_dlg = args.run_dlg
    cfg.attack_seed = args.attack_seed
    cfg.mia_sample_size = args.mia_sample_size
    cfg.mia_bootstrap_samples = args.mia_bootstrap_samples
    cfg.run_ilrg = args.run_ilrg
    cfg.ilrg_batch_size = (
        args.ilrg_batch_size if args.ilrg_batch_size is not None
        else args.local_batch_size
    )
    cfg.ilrg_num_batches = args.ilrg_num_batches
    cfg.ilrg_alpha = args.ilrg_alpha
    cfg.ilrg_mask_mode = args.ilrg_mask_mode
    cfg.dlg_num_samples = args.dlg_num_samples
    cfg.dlg_num_restarts = args.dlg_num_restarts
    cfg.dlg_iterations = args.dlg_iterations
    cfg.dlg_learning_rate = args.dlg_learning_rate
    cfg.dlg_optimizer = args.dlg_optimizer
    cfg.dlg_objective = args.dlg_objective
    cfg.dlg_tv_weight = args.dlg_tv_weight
    cfg.dlg_early_stopping_patience = args.dlg_early_stopping_patience
    cfg.dlg_success_ssim = args.dlg_success_ssim
    cfg.dlg_compute_lpips = args.dlg_compute_lpips
    cfg.dlg_known_label = args.dlg_known_label
    cfg.run_ig = args.run_ig
    cfg.ig_num_samples = args.ig_num_samples
    cfg.ig_num_restarts = args.ig_num_restarts
    cfg.ig_iterations = args.ig_iterations
    cfg.ig_learning_rate = args.ig_learning_rate
    cfg.ig_tv_weight = args.ig_tv_weight
    cfg.ig_early_stopping_patience = args.ig_early_stopping_patience
    cfg.ig_success_ssim = args.ig_success_ssim
    cfg.ig_compute_lpips = args.ig_compute_lpips
    cfg.ig_known_label = args.ig_known_label

    if cfg.attack_interval <= 0:
        raise ValueError("attack_interval must be positive")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if cfg.mask_gradient_batch_size <= 0:
        raise ValueError("mask_gradient_batch_size must be positive")
    if not 0.0 <= cfg.encryption_ratio <= 1.0:
        raise ValueError("encryption_ratio must be between 0 and 1")
    if cfg.mia_sample_size <= 0:
        raise ValueError("mia_sample_size must be positive")
    if cfg.mia_bootstrap_samples < 0:
        raise ValueError("mia_bootstrap_samples cannot be negative")
    if cfg.ilrg_batch_size <= 0:
        raise ValueError("ilrg_batch_size must be positive")
    if cfg.ilrg_num_batches <= 0:
        raise ValueError("ilrg_num_batches must be positive")
    if cfg.ilrg_alpha <= 0:
        raise ValueError("ilrg_alpha must be positive")
    if cfg.dlg_num_samples <= 0:
        raise ValueError("dlg_num_samples must be positive")
    if cfg.dlg_num_restarts <= 0:
        raise ValueError("dlg_num_restarts must be positive")
    if cfg.dlg_iterations <= 0:
        raise ValueError("dlg_iterations must be positive")
    if cfg.dlg_learning_rate <= 0:
        raise ValueError("dlg_learning_rate must be positive")
    if cfg.dlg_tv_weight < 0:
        raise ValueError("dlg_tv_weight cannot be negative")
    if cfg.dlg_early_stopping_patience < 0:
        raise ValueError("dlg_early_stopping_patience cannot be negative")
    if not -1.0 <= cfg.dlg_success_ssim <= 1.0:
        raise ValueError("dlg_success_ssim must be between -1 and 1")
    if cfg.ig_num_samples <= 0:
        raise ValueError("ig_num_samples must be positive")
    if cfg.ig_num_restarts <= 0:
        raise ValueError("ig_num_restarts must be positive")
    if cfg.ig_iterations <= 0:
        raise ValueError("ig_iterations must be positive")
    if cfg.ig_learning_rate <= 0:
        raise ValueError("ig_learning_rate must be positive")
    if cfg.ig_tv_weight < 0:
        raise ValueError("ig_tv_weight cannot be negative")
    if cfg.ig_early_stopping_patience < 0:
        raise ValueError("ig_early_stopping_patience cannot be negative")
    if not -1.0 <= cfg.ig_success_ssim <= 1.0:
        raise ValueError("ig_success_ssim must be between -1 and 1")

    cfg.configString = (f'{cfg.moduleName}_{cfg.model.name}_'
                        f'{cfg.DB_dataset.name}_seed{cfg.seed}_'
                        f'attack{int(cfg.attack)}_interval{cfg.attack_interval}'
                        f'KP{str(cfg.encryption_ratio)}_'
                        f'alpha{cfg.DB_nonIID_alpha}_DataS{cfg.DB_samples_per_client / 1000.0:.1f}k'
                        f'_R{cfg.rounds}_E{cfg.local_epochs}_C{cfg.num_clients}_BN{int(cfg.aggregate_BN)}')
    if cfg.exp_id is None:
        cfg.exp_id = cfg.configString
    cfg.run_id = (
        safe_path_component(args.run_id)
        if args.run_id is not None
        else generate_run_id(cfg.exp_id or cfg.configString)
    )
    cfg.resolved_args = vars(args).copy()
    # set seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32

    # print
    print("Parsed Arguments:")
    print("exp_group: ", cfg.group)
    print(f"num_clients: {args.num_clients}")
    print(f"encryption_ratio: {args.encryption_ratio}")
    print(f"alpha: {cfg.DB_nonIID_alpha}")
    print(f"local_epochs: {args.local_epochs}")
    print(f"rounds: {args.rounds}")
    print(f"seed: {args.seed}")
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print(f"exp_id: {args.exp_id}")
    print(f"local_batch_size: {args.local_batch_size}")
    print(f"eval_batch_size: {args.eval_batch_size}")
    print(f"mask_gradient_batch_size: {args.mask_gradient_batch_size}")
    print(f"tf32: {cfg.tf32} (effective only on supported CUDA GPUs)")
    print(f"temperature: {args.temperature}")
    print(f"moduleName: {cfg.moduleName}")
    print(f"DB_samples_per_client: {cfg.DB_samples_per_client}")
    print(f'{cfg.model.name}_{cfg.moduleName}_')
    print(f'{cfg.DB_dataset.name}_seed{cfg.seed}_')
    print(f'attack{bool(cfg.attack)}_')
    print(f'interval{cfg.attack_interval}_')
    print(
        f"run_mia: {cfg.run_mia}, run_ilrg: {cfg.run_ilrg}, run_dlg: {cfg.run_dlg}, "
        f"run_ig: {cfg.run_ig}, attack_seed: {cfg.attack_seed}"
    )
    print(
        f"iLRG: {cfg.ilrg_num_batches} batches x up to {cfg.ilrg_batch_size} samples "
        f"(alpha={cfg.ilrg_alpha}, mask_mode={cfg.ilrg_mask_mode})"
    )
    print(
        f"DLG: {cfg.dlg_num_samples} samples x {cfg.dlg_num_restarts} restarts x "
        f"{cfg.dlg_iterations} iterations ({cfg.dlg_optimizer}, {cfg.dlg_objective}, "
        f"known_label={cfg.dlg_known_label})"
    )
    print(
        f"IG: {cfg.ig_num_samples} samples x {cfg.ig_num_restarts} restarts x "
        f"{cfg.ig_iterations} iterations (signed Adam, global cosine, "
        f"early_stopping_patience={cfg.ig_early_stopping_patience}, "
        f"known_label={cfg.ig_known_label})"
    )
    print(f'KP{str(cfg.encryption_ratio)}_')
    print(f'_R{cfg.rounds}_E{cfg.local_epochs}_C{cfg.num_clients}')

    return cfg

cfg = build_config()
# ############################################ Prepare Setup ############################################
foldersInit(cfg=cfg)
sep = "/"
tempDir_addr = "./Temp/" + cfg.model.name + sep + cfg.run_id + sep
os.makedirs(tempDir_addr, exist_ok=True)
delete_all_in_folder(tempDir_addr)
ckks = HE_CKKS(createNew=True, baseAddr=tempDir_addr + "CKKS/")

# ############################################ Prepare Dataset ############################################
dataset = DatasetLoader(dataset_type=cfg.DB_dataset, data_dir=cfg.dataSetAddress, seed=cfg.seed)
dataset.load_and_preprocess(force_reload=cfg.DB_forceCreate)
cfg.classNum, cfg.inputShape = dataset.num_classes, dataset.input_shape
client_data, client_labels = dataset.partition_data(
    num_clients=cfg.num_clients,
    samples_per_client=cfg.num_clients * [cfg.DB_samples_per_client],
    alpha=cfg.DB_nonIID_alpha,
    force_create=cfg.DB_forceCreate,
    is_iid=not cfg.DB_nonIID
)
testData = dataset.get_test_data()

excelDir = Path("./Results/Accuracies") / cfg.model.name / cfg.group / cfg.run_id
excelDir.mkdir(parents=True, exist_ok=True)

# ############################################ Define Exposed Model ############################################
trainer = TrainEngine(cfg=cfg, testData=testData, globalData=None)
clientTensorData = [
    trainer.prepare_device_tensors(images, labels)
    for images, labels in zip(client_data, client_labels)
]
client_dataset_sizes = np.asarray([len(images) for images in client_data], dtype=np.int64)
if len(client_dataset_sizes) != cfg.num_clients or np.any(client_dataset_sizes <= 0):
    raise ValueError("Every configured client must have a non-empty data partition")
client_weights = client_dataset_sizes.astype(np.float64)
client_weights /= client_weights.sum()
clientTensorBytes = sum(
    tensor.numel() * tensor.element_size()
    for client_tensors in clientTensorData
    for tensor in client_tensors
)
print(
    f"Cached all client partitions on {trainer.device}: "
    f"{clientTensorBytes / (1024 ** 2):.1f} MiB"
)
exposedModel = trainer.getAllWeights()         # list of numpy arrays
exposed_flat = flattener(exposedModel).astype(np.float32, copy=False)
totalModelWeightCount = len(exposed_flat)

modelStructure = extractStructure(exposedModel)
print(f"Number of layers of the model: {len(modelStructure)}")
print(f"Total count of weights of the model: {sum(layer['size'] for layer in modelStructure)}")

checkpoint_writer = None
if cfg.attack:
    checkpoint_writer = AttackCheckpointWriter(
        base_dir=Path(cfg.checkpoint_root),
        project_root=Path(__file__).resolve().parent,
        cfg=cfg,
        model=trainer.model,
        initial_state=exposedModel,
        client_data=client_data,
        client_labels=client_labels,
        test_data=testData,
        dataset_train_data=(dataset.X_train, dataset.y_train),
    )
    print(f"Offline attack checkpoint run: {checkpoint_writer.run_dir}")

# ############################################ Mask Generation Phase ############################################
print("\n" + "#" * 80)
print("GENERATING FEDML-HE ENCRYPTION MASK")
print("#" * 80 + "\n")
cfg.currentRound = -1
cfg.excelAddr = str(excelDir / f"MaskGen_{cfg.configString}")
encrypted_sensitivity_maps = []
plaintext_sensitivity_maps = []
size_client_sensitivityMaps = []
time_ClientMaskProposal = 0.0
excelHelper.create(cfg=cfg)

# Algorithm 1, "Local Sensitivity Map Calculation".  Every client evaluates
# the same initial model, so the resulting maps differ only because of local
# data.  This phase runs once, before federated training.
for client_idx in range(cfg.num_clients):
    cfg.currentEdge = client_idx
    print(f"  Calculating sensitivity map for client {client_idx}")
    print(40 * "/^\\")
    cfg.saveAddr = tempDir_addr + f"MaskGen_c{client_idx}"
    trainer.setAllWeights(weights=exposedModel)
    t0 = time.perf_counter()
    sensitivity_map = trainer.proposeMask(data=clientTensorData[client_idx])
    encrypted_sensitivity_maps.append(ckks.encrypt_data(sensitivity_map))
    time_ClientMaskProposal += time.perf_counter() - t0
    plaintext_sensitivity_maps.append([np.asarray(layer).copy() for layer in sensitivity_map])
    size_client_sensitivityMaps.append(ckks.measure_encrypted_size(encrypted_sensitivity_maps[-1]))

# Clients operate in parallel in the protocol, so report mean client latency
# rather than the sum of all simulated client runtimes.
time_ClientMaskProposal /= cfg.num_clients

# Algorithm 1, "Server Encryption Mask Aggregation".  Sensitivity maps remain
# encrypted while the server applies the FedAvg data-size weights.
t0 = time.perf_counter()
aggregated_encrypted_sensitivity = ckks.encrypted_multiply_scalar(encrypted_sensitivity_maps[0], client_weights[0])
for client_idx in range(1, cfg.num_clients):
    weighted_sensitivity = ckks.encrypted_multiply_scalar(encrypted_sensitivity_maps[client_idx], client_weights[client_idx])
    aggregated_encrypted_sensitivity = ckks.encrypted_add(aggregated_encrypted_sensitivity, weighted_sensitivity)

time_sensitivityMapsAggregation = time.perf_counter() - t0
size_server_sensitivity_map = ckks.measure_encrypted_size(aggregated_encrypted_sensitivity)

# In the single-key simulation, an authorized client decrypts the aggregated
# privacy map and selects the configured fraction of most-sensitive trainable
# parameters. Each client performs the same work locally; this simulation runs
# it once and reports that representative per-client latency. Non-trainable
# state is mapped explicitly for model transport.
t0 = time.perf_counter()
aggregated_sensitivity = ckks.decrypt_data(aggregated_encrypted_sensitivity)    # decrypt the encrypted aggregated sensitivity map received from server
trainable_mask = maskFilter(aggregated_sensitivity, cfg.encryption_ratio)       # select the top p
globalMask = trainer.mapTrainableToAllVars(trainable_mask)                      # include BN indices
maskBool, maskBoolNot = indexMask_to_BinaryMask(globalMask, modelStructure)     # generate boolean mask from index map
maskPlan = build_mask_plan(maskBool, maskBoolNot)
plaintext_indices = flattener(maskBoolNot).astype(bool, copy=False)
time_MaskDecryption = time.perf_counter() - t0
assert plaintext_indices.shape == exposed_flat.shape
encrypted_parameter_count = (totalModelWeightCount if globalMask is None else int(len(globalMask)))

time_MaskGen = time_ClientMaskProposal + time_sensitivityMapsAggregation + time_MaskDecryption
print(  f"  Global mask encrypts {encrypted_parameter_count}/{totalModelWeightCount} "
        f"parameters ({encrypted_parameter_count / totalModelWeightCount * 100:.2f}%; "
        f"configured trainable ratio: {cfg.encryption_ratio * 100:.2f}%).")

# ############################################ Begin Federated Learning ############################################
print("\n" + "#" * 80)
print("STARTING FEDERATED LEARNING")
print("#" * 80 + "\n")
BN = [None] * cfg.num_clients

for round in range(cfg.rounds):
    # ______________ preparation ______________
    cfg.currentRound = round
    print("\n" + "=" * 80)
    print(f"FEDERATED LEARNING ROUND {round}/{cfg.rounds}")
    print("=" * 80)
    cfg.excelAddr = str(excelDir / f"round_{round}")
    cfg.saveAddr_iter = tempDir_addr + str(round) + "Iter_"
    excelHelper.create(cfg=cfg)
    filtered_dnn_handler = FilteredDNNEncryption(cfg=cfg, ckks_instance=ckks, save_dir=cfg.saveAddr_iter)

    # ______________ Load global model ______________
    if round == 0:
        time_reconstructing = -1    # -1 means no reconstruction happened in round 0
        globalModel = exposedModel
    else:
        globalModel = aggregatedModel

    # ______________ train all clients ______________
    clientsModels  = []

    for client_idx in range(cfg.num_clients):
        # ______________ preparation ______________
        cfg.currentEdge = client_idx
        print(f"\n[Round {round}] (Training Client {client_idx}/{cfg.num_clients})")
        print(40 * "-")
        cfg.saveAddr = f"{cfg.saveAddr_iter}(edge_{client_idx})_{cfg.local_epochs}E"

        # ______________ train this client ______________
        trainer.setAllWeights(weights=globalModel)
        if not cfg.aggregate_BN and round != 0:
            trainer.setBN(weights=BN[client_idx])   # restore this client's own BN stats from its previous round

        trainer.train(trainData=clientTensorData[client_idx])
        updatedWeights = trainer.getAllWeights(saveWeights=False)
        clientsModels.append(updatedWeights)

        if not cfg.aggregate_BN:
            BN[client_idx] = trainer.getBN()        # save this client's local BN stats for its next round

    # if round == 0: # shows which weights are encrypted
    #     filtered_dnn_handler.analyze_encryption_criticality(maskBool)

    # ______________ Send updated models to server ______________
    clientSentModels = []
    for client_idx in range(cfg.num_clients):
        cfg.currentEdge = client_idx
        clientSentModel, _ = filtered_dnn_handler.filteringDNN(clientsModels[client_idx], maskPlan)
        clientSentModels.append(clientSentModel)
        print("Client", client_idx, "sent its trained model to server.")

        if round == 0:
            # Mask generation is a one-time initialization cost, recorded
            # alongside the other round-0 payload sizes.
            excelHelper.update(cfg=cfg, dataDic={
                "size_sensitivity_map": size_client_sensitivityMaps[client_idx],
            })

    should_capture_round = (
        checkpoint_writer is not None
        and (round % cfg.attack_interval == 0 or round == cfg.rounds - 1)
    )
    if should_capture_round:
        transmission_metadata = [
            {
                "logical_payload": "aligned encrypted and plaintext model coordinates",
                "ciphertext_serialized": False,
                "encrypted_payload_bytes": int(sent.get("size_cyphertext", 0)),
                "plaintext_payload_bytes": int(sent.get("size_plaintext", 0)),
                "transport_ciphertext_extension_point": None,
            }
            for sent in clientSentModels
        ]
        checkpoint_copy_seconds = checkpoint_writer.prepare_round(
            round_index=round,
            global_entering=globalModel,
            exposed_flat_before_update=exposed_flat,
            encrypted_masks=maskBool,
            plaintext_masks=maskBoolNot,
            global_mask=globalMask,
            mask_plan=maskPlan,
            # Stored under evaluation-oracle metadata because these plaintext
            # client maps are unavailable to the honest-but-curious server.
            sensitivity_maps=plaintext_sensitivity_maps,
            client_states=clientsModels,
            client_weights=client_weights,
            client_dataset_sizes=client_dataset_sizes,
            transmission_metadata=transmission_metadata,
        )
        print(
            f"  Captured immutable pre-aggregation attack state in "
            f"{checkpoint_copy_seconds:.3f}s"
        )

    # ______________ server side aggregation ______________
    print("Server begins aggregation")
    t0 = time.perf_counter()
    aggregated = filtered_dnn_handler.aggregate_filtered_dnns(             # includes metadata
        clientSentModels,
        weights=client_weights.astype(np.float32, copy=False)
    )
    time_aggregation = time.perf_counter() - t0
    excelHelper.update(cfg=cfg, dataDic={
        "time_aggregation": time_aggregation,
        "time_reconstructing": time_reconstructing,
        # Mask generation is a one-time initialization cost, recorded in round
        # zero so summing round metrics does not count it repeatedly.
        "time_ClientMaskProposal": time_ClientMaskProposal if round == 0 else 0.0,      # averaged over all clients, in this round
        "time_sensitivityMapsAggregation": time_sensitivityMapsAggregation if round == 0 else 0.0,
        "time_MaskDecryption": time_MaskDecryption if round == 0 else 0.0,
        "time_MaskGen": time_MaskGen if round == 0 else 0.0,
        "size_server_sensitivity_map": size_server_sensitivity_map if round == 0 else 0,

        "totalSize_encrypted": aggregated['totalSize_encrypted'],
        "totalSize_plaintext": aggregated['totalSize_plaintext'],
        "totalSize_Sum": aggregated['totalSize_Sum'],
        "size_server_cyphertext": aggregated['size_server_cyphertext'],
        "size_server_plaintext": aggregated['size_server_plaintext'],
    })
    print(f"  Aggregation complete in {time_aggregation:.2f}s")

    # ____________________________ Update ExposedModel ____________________________
    t0 = time.perf_counter()
    aggregatedModel = filtered_dnn_handler.reconstructDNN(aggregated)
    time_reconstructing = time.perf_counter() - t0
    aggregated_flat = flattener(aggregatedModel)
    assert plaintext_indices.shape == exposed_flat.shape == aggregated_flat.shape
    exposed_flat[plaintext_indices] = aggregated_flat[plaintext_indices]
    print(f"  Exposed model updated. {np.sum(plaintext_indices)} parameters updated.")

    if should_capture_round:
        checkpoint_round_manifest = checkpoint_writer.finalize_round(
            global_after_aggregation=aggregatedModel,
            exposed_flat_after_update=exposed_flat,
            post_aggregation_metadata={
                "timing": "after aggregation and round-ending exposed-state update",
                "totalSize_encrypted": int(aggregated["totalSize_encrypted"]),
                "totalSize_plaintext": int(aggregated["totalSize_plaintext"]),
                "totalSize_Sum": int(aggregated["totalSize_Sum"]),
                "size_server_cyphertext": int(aggregated["size_server_cyphertext"]),
                "size_server_plaintext": int(aggregated["size_server_plaintext"]),
            },
        )
        checkpoint_seconds = checkpoint_round_manifest["timing"][
            "total_checkpoint_seconds"
        ]
        excelHelper.update(cfg=cfg, dataDic={"time_checkpoint": checkpoint_seconds})
        print(
            f"  Offline attack checkpoint committed in {checkpoint_seconds:.3f}s "
            f"({checkpoint_round_manifest['storage_bytes'] / (1024 ** 2):.1f} MiB)"
        )

    # ======================= Evaluate global model =======================
    print(f"\n[Round {round}] Evaluating global model...")
    trainer.setAllWeights(weights=aggregatedModel)
    test_acc = trainer.evaluate_test()
    print(f"  Global Test Accuracy: {test_acc:.4f}")
    excelHelper.update(cfg=cfg, dataDic={"global_test_acc": test_acc})
    excelHelper.flush(cfg)
    # ============================================================

############################## Combine Excel Results ####################################
resultExcelAddr = str(excelDir / "Final")
savedExcelsAddr = str(excelDir / "round_%d")

sheet_list = ["Test Accuracy MAX", "Train Accuracy MAX",
              "Test Accuracy", "Train Accuracy", 'global_test_acc',
              "sensitivityScore", "Times",
              'size_plaintext', 'size_cyphertext', 'size_sensitivity_map']

excelHelper.combineExcels(cfg=cfg, baseAddr=savedExcelsAddr, targetSaveAddr=resultExcelAddr, sheetList=sheet_list)

if checkpoint_writer is not None:
    checkpoint_writer.finalize_experiment()
    print(f"Completed offline attack checkpoint: {checkpoint_writer.run_dir}")

print("\n" + "#" * 80)
print("FEDERATED LEARNING COMPLETE")
print("#" * 80 + "\n")

# Clean up
# No session to close in PyTorch; kept for symmetry with original structure.
del trainer
gc.collect()
delete_all_in_folder(tempDir_addr)
