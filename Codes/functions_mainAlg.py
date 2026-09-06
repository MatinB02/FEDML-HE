import time
import numpy as np
from Codes import excelHelper
from Codes.CKKSRun import HE_CKKS
import pickle
import os
import torch  # <-- Imported to safely parse incoming model updates from Flower/PyTorch


def build_mask_plan(mask, mask_not):
    """Precompute the shared packing layout for one round's consensus mask."""
    if mask is None or mask_not is None:
        raise ValueError("mask and mask_not are required to build a mask plan")
    if len(mask) != len(mask_not):
        raise ValueError("mask and mask_not must have the same number of layers")

    layer_shapes = []
    encrypted_indices = []
    plaintext_indices = []

    for layer_idx, (layer_mask, layer_mask_not) in enumerate(zip(mask, mask_not)):
        layer_mask = np.asarray(layer_mask)
        layer_mask_not = np.asarray(layer_mask_not)
        if layer_mask.shape != layer_mask_not.shape:
            raise ValueError(
                f"Layer {layer_idx}: mask shape mismatch "
                f"mask={layer_mask.shape}, mask_not={layer_mask_not.shape}"
            )
        encrypted = np.flatnonzero(layer_mask).astype(np.int64, copy=False)
        plaintext = np.flatnonzero(layer_mask_not).astype(np.int64, copy=False)
        encrypted.setflags(write=False)
        plaintext.setflags(write=False)

        layer_shapes.append(tuple(layer_mask.shape))
        encrypted_indices.append(encrypted)
        plaintext_indices.append(plaintext)

    encrypted_boundaries = np.cumsum(
        [0] + [indices.size for indices in encrypted_indices], dtype=np.int64
    ).tolist()
    plaintext_boundaries = np.cumsum(
        [0] + [indices.size for indices in plaintext_indices], dtype=np.int64
    ).tolist()

    metadata = {
        'num_layers': len(layer_shapes),
        'layer_shapes': layer_shapes,
        'encrypted_info': [
            {
                'layer_idx': i,
                'num_values': int(indices.size),
                'total_elements': int(np.prod(layer_shapes[i])),
                'sparsity': float(
                    1.0 - indices.size / max(int(np.prod(layer_shapes[i])), 1)
                ),
            }
            for i, indices in enumerate(encrypted_indices)
        ],
        'plaintext_info': [
            {
                'layer_idx': i,
                'num_values': int(indices.size),
                'total_elements': int(np.prod(layer_shapes[i])),
                'sparsity': float(
                    1.0 - indices.size / max(int(np.prod(layer_shapes[i])), 1)
                ),
            }
            for i, indices in enumerate(plaintext_indices)
        ],
        'encrypted_layer_boundaries': encrypted_boundaries,
        'plaintext_layer_boundaries': plaintext_boundaries,
    }

    return {
        'layer_shapes': tuple(layer_shapes),
        'encrypted_indices': tuple(encrypted_indices),
        'plaintext_indices': tuple(plaintext_indices),
        'encrypted_layer_boundaries': tuple(encrypted_boundaries),
        'plaintext_layer_boundaries': tuple(plaintext_boundaries),
        'encrypted_weights_sparse': tuple(
            {'indices': indices} for indices in encrypted_indices
        ),
        'plaintext_weights_sparse': tuple(
            {'indices': indices} for indices in plaintext_indices
        ),
        'metadata': metadata,
    }


class FilteredDNNEncryption:
    def __init__(self, cfg, ckks_instance=None, save_dir="./filtered_dnn"):
        self.cfg = cfg
        self.ckks = ckks_instance if ckks_instance else HE_CKKS(createNew=True)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _empty_sparse(num_layers):
        return [{'indices': np.array([], dtype=np.int64)}
                for _ in range(num_layers)]

    def filteringDNN(self, dnn_weights, mask_plan, save_path=None):
        if dnn_weights is None or len(dnn_weights) == 0:
            result = {
                'encrypted_data': None,
                'plaintext_data': None,
                'encrypted_weights_sparse': [],
                'plaintext_weights_sparse': [],
                'size_cyphertext': 0,
                'size_plaintext': 0,
                'metadata': {
                    'num_layers': 0,
                    'layer_shapes': [],
                    'encrypted_info': [],
                    'plaintext_info': [],
                    'encrypted_layer_boundaries': [0],
                    'plaintext_layer_boundaries': [0],
                }
            }
            return result, 0.0

        # --- PYTORCH COMPATIBILITY LAYER ---
        # If weights are passed from PyTorch/Flower as Tensors, clean convert them to NumPy arrays.
        cleaned_weights = []
        for w in dnn_weights:
            if isinstance(w, torch.Tensor):
                cleaned_weights.append(w.detach().cpu().numpy().astype(np.float64))
            else:
                cleaned_weights.append(w)
        dnn_weights = cleaned_weights
        # -----------------------------------

        layer_shapes = mask_plan['layer_shapes']
        if len(dnn_weights) != len(layer_shapes):
            raise ValueError("dnn_weights and mask plan must have the same number of layers")
        for i, (weights, expected_shape) in enumerate(zip(dnn_weights, layer_shapes)):
            if weights.shape != expected_shape:
                raise ValueError(
                    f"Layer {i}: shape mismatch "
                    f"weights={weights.shape}, mask_plan={expected_shape}"
                )

        maskingTime = 0.0
        encryptionTime = 0.0
        metadata = mask_plan['metadata']
        encrypted_weights_sparse = mask_plan['encrypted_weights_sparse']
        plaintext_weights_sparse = mask_plan['plaintext_weights_sparse']
        encrypted_layer_boundaries = mask_plan['encrypted_layer_boundaries']
        plaintext_layer_boundaries = mask_plan['plaintext_layer_boundaries']

        t0 = time.perf_counter()
        encrypted_values_array = np.empty(encrypted_layer_boundaries[-1], dtype=np.float64)
        plaintext_values_array = np.empty(plaintext_layer_boundaries[-1], dtype=np.float64)
        for i, weights in enumerate(dnn_weights):
            weights_flat = weights.reshape(-1)
            enc_start, enc_end = encrypted_layer_boundaries[i:i + 2]
            plain_start, plain_end = plaintext_layer_boundaries[i:i + 2]
            encrypted_values_array[enc_start:enc_end] = weights_flat[
                mask_plan['encrypted_indices'][i]
            ]
            plaintext_values_array[plain_start:plain_end] = weights_flat[
                mask_plan['plaintext_indices'][i]
            ]
        maskingTime += time.perf_counter() - t0

        if encrypted_layer_boundaries[-1] > 0:
            dummy_model = [encrypted_values_array]

            t0 = time.perf_counter()
            # all_encrypted_values is already the consensus-mask-selected sequence.
            # Keep it dense so a client's private zero values occupy their agreed
            # slots instead of becoming cleartext, value-dependent layout metadata.
            encrypted_data = self.ckks.encrypt_data(dummy_model, mask=None)
            encryptionTime += (time.perf_counter() - t0)
            del encrypted_values_array, dummy_model
        else:
            encrypted_data = None
            print("No values to encrypt (encrypted_data=None).")

        if plaintext_layer_boundaries[-1] > 0:
            dummy_model = [plaintext_values_array]

            t0 = time.perf_counter()
            plaintext_data = self.ckks.encode_data(dummy_model, mask=None)
            encryptionTime += (time.perf_counter() - t0)
            del plaintext_values_array, dummy_model
        else:
            plaintext_data = None
            print("No values to encode as plaintext (plaintext_data=None).")

        result = {
            'encrypted_data': encrypted_data,
            'plaintext_data': plaintext_data,
            'encrypted_weights_sparse': encrypted_weights_sparse,
            'plaintext_weights_sparse': plaintext_weights_sparse,
            'metadata': metadata,
            # Fresh client payload sizes, measured from the serialized objects
            # produced above (metadata and model reconstruction indices excluded,
            # consistently with the existing server payload measurements).
            'size_cyphertext': (
                encrypted_data.get('total_bytes', 0) if encrypted_data is not None else 0
            ),
            'size_plaintext': (
                plaintext_data.get('total_bytes', 0) if plaintext_data is not None else 0
            ),
        }

        if save_path:
            self.save_filtered_dnn(result, save_path)

        try:
            excelHelper.update(cfg=self.cfg, dataDic={
                "time_masking": maskingTime,
                "time_encryption": encryptionTime,
                "size_cyphertext": result['size_cyphertext'],
                "size_plaintext": result['size_plaintext'],
            })
        except Exception as e:
            print(f"⚠ excelHelper.update failed (ignored): {e}")

        return result, maskingTime

    def save_filtered_dnn(self, filtered_dnn_dict, save_path):
        base_path = os.path.join(self.save_dir, save_path)
        os.makedirs(base_path, exist_ok=True)

        if filtered_dnn_dict.get('encrypted_data') is not None:
            encrypted_path = os.path.join(base_path, "encrypted")
            self.ckks.save_encrypted_data(filtered_dnn_dict['encrypted_data'], encrypted_path)
        else:
            print("  - No encrypted data to save")

        if filtered_dnn_dict.get('plaintext_data') is not None:
            plaintext_path = os.path.join(base_path, "plaintext_encoded")
            self.ckks.save_plaintext_data(filtered_dnn_dict['plaintext_data'], plaintext_path)
        else:
            print("  - No plaintext encoded data to save")

        enc_sparse = filtered_dnn_dict.get('encrypted_weights_sparse', [])
        pt_sparse = filtered_dnn_dict.get('plaintext_weights_sparse', [])

        encrypted_sparse_path = os.path.join(base_path, "encrypted_sparse.npz")
        encrypted_sparse_dict = {}
        for i, sparse_data in enumerate(enc_sparse):
            encrypted_sparse_dict[f'indices_{i}'] = sparse_data.get('indices', np.array([], dtype=np.int64))
        np.savez_compressed(encrypted_sparse_path, **encrypted_sparse_dict)

        plaintext_sparse_path = os.path.join(base_path, "plaintext_sparse.npz")
        plaintext_sparse_dict = {}
        for i, sparse_data in enumerate(pt_sparse):
            plaintext_sparse_dict[f'indices_{i}'] = sparse_data.get('indices', np.array([], dtype=np.int64))
        np.savez_compressed(plaintext_sparse_path, **plaintext_sparse_dict)

        metadata_path = os.path.join(base_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(filtered_dnn_dict.get('metadata', {}), f)

    def load_filtered_dnn(self, save_path):
        base_path = os.path.join(self.save_dir, save_path)
        metadata_path = os.path.join(base_path, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            metadata = {'num_layers': 0, 'layer_shapes': [],
                        'encrypted_layer_boundaries': [0],
                        'plaintext_layer_boundaries': [0]}
            print("  ⚠ Metadata not found; using empty defaults")

        num_layers = int(metadata.get('num_layers', 0))

        encrypted_path = os.path.join(base_path, "encrypted")
        if os.path.exists(encrypted_path + "_metadata.pkl"):
            encrypted_data = self.ckks.load_encrypted_data(encrypted_path)
        else:
            encrypted_data = None
            print("  - No encrypted data found")

        plaintext_path = os.path.join(base_path, "plaintext_encoded")
        if os.path.exists(plaintext_path + "_metadata.pkl"):
            plaintext_data = self.ckks.load_plaintext_data(plaintext_path)
        else:
            plaintext_data = None
            print("  - No plaintext encoded data found")

        encrypted_weights_sparse = self._empty_sparse(num_layers)
        plaintext_weights_sparse = self._empty_sparse(num_layers)

        encrypted_sparse_path = os.path.join(base_path, "encrypted_sparse.npz")
        if os.path.exists(encrypted_sparse_path):
            encrypted_sparse_npz = np.load(encrypted_sparse_path)
            for i in range(num_layers):
                key = f'indices_{i}'
                if key in encrypted_sparse_npz.files:
                    encrypted_weights_sparse[i]['indices'] = encrypted_sparse_npz[key]
        else:
            print("  - Encrypted indices file missing; using empty indices")

        plaintext_sparse_path = os.path.join(base_path, "plaintext_sparse.npz")
        if os.path.exists(plaintext_sparse_path):
            plaintext_sparse_npz = np.load(plaintext_sparse_path)
            for i in range(num_layers):
                key = f'indices_{i}'
                if key in plaintext_sparse_npz.files:
                    plaintext_weights_sparse[i]['indices'] = plaintext_sparse_npz[key]
        else:
            print("  - Plaintext indices file missing; using empty indices")

        return {
            'encrypted_data': encrypted_data,
            'plaintext_data': plaintext_data,
            'encrypted_weights_sparse': encrypted_weights_sparse,
            'plaintext_weights_sparse': plaintext_weights_sparse,
            'metadata': metadata
        }

    def reconstructDNN(self, filtered_dnn_dict, fill_encrypted='decrypt', rng=None):
        print("\nReconstructing DNN from filtered data...")

        if fill_encrypted not in ('decrypt', 'zeros', 'random'):
            raise ValueError("fill_encrypted must be 'decrypt', 'zeros', or 'random'")

        if rng is None and fill_encrypted == 'random':
            rng = np.random.default_rng()

        metadata = filtered_dnn_dict.get('metadata', {})
        num_layers = int(metadata.get('num_layers', 0))
        layer_shapes = metadata.get('layer_shapes', [])

        encrypted_weights_sparse = filtered_dnn_dict.get('encrypted_weights_sparse', self._empty_sparse(num_layers))
        plaintext_weights_sparse = filtered_dnn_dict.get('plaintext_weights_sparse', self._empty_sparse(num_layers))

        encrypted_layer_boundaries = metadata.get('encrypted_layer_boundaries', [0] * (num_layers + 1))
        plaintext_layer_boundaries = metadata.get('plaintext_layer_boundaries', [0] * (num_layers + 1))

        decrypted_by_layer = []
        if fill_encrypted == 'decrypt':
            if filtered_dnn_dict.get('encrypted_data') is not None:
                decrypted_model = self.ckks.decrypt_data(filtered_dnn_dict['encrypted_data'])
                all_decrypted_values = np.array(decrypted_model[0], dtype=np.float64)
            else:
                print("  No encrypted_data found; using zeros for encrypted region.")
                all_decrypted_values = np.array([], dtype=np.float64)

            if len(encrypted_layer_boundaries) == num_layers + 1:
                for i in range(num_layers):
                    s = int(encrypted_layer_boundaries[i])
                    e = int(encrypted_layer_boundaries[i + 1])
                    decrypted_by_layer.append(all_decrypted_values[s:e])
            else:
                for i in range(num_layers):
                    n = len(encrypted_weights_sparse[i].get('indices', []))
                    decrypted_by_layer.append(np.zeros(n, dtype=np.float64))

        else:
            for i in range(num_layers):
                n = len(encrypted_weights_sparse[i].get('indices', []))
                if fill_encrypted == 'zeros':
                    decrypted_by_layer.append(np.zeros(n, dtype=np.float64))
                else:
                    decrypted_by_layer.append(rng.normal(0.0, 0.01, size=n).astype(np.float64))

        if filtered_dnn_dict.get('plaintext_data') is not None:
            decoded_model = self.ckks.decode_data(filtered_dnn_dict['plaintext_data'])
            all_decoded_values = np.array(decoded_model[0], dtype=np.float64)
        else:
            print("  No plaintext_data found; using zeros for plaintext region.")
            all_decoded_values = np.array([], dtype=np.float64)

        decoded_by_layer = []
        if len(plaintext_layer_boundaries) == num_layers + 1:
            for i in range(num_layers):
                s = int(plaintext_layer_boundaries[i])
                e = int(plaintext_layer_boundaries[i + 1])
                decoded_by_layer.append(all_decoded_values[s:e])
        else:
            for i in range(num_layers):
                n = len(plaintext_weights_sparse[i].get('indices', []))
                decoded_by_layer.append(np.zeros(n, dtype=np.float64))

        reconstructed_dnn = []
        for i in range(num_layers):
            layer_shape = tuple(layer_shapes[i])
            total_elements = int(np.prod(layer_shape))

            reconstructed_flat = np.zeros(total_elements, dtype=np.float64)

            enc_idx = encrypted_weights_sparse[i].get('indices', np.array([], dtype=np.int64))
            enc_val = decrypted_by_layer[i] if i < len(decrypted_by_layer) else np.array([], dtype=np.float64)
            m = min(len(enc_idx), len(enc_val))
            if m > 0:
                reconstructed_flat[enc_idx[:m]] = enc_val[:m]

            pt_idx = plaintext_weights_sparse[i].get('indices', np.array([], dtype=np.int64))
            pt_val = decoded_by_layer[i] if i < len(decoded_by_layer) else np.array([], dtype=np.float64)
            m = min(len(pt_idx), len(pt_val))
            if m > 0:
                reconstructed_flat[pt_idx[:m]] = pt_val[:m]

            reconstructed_layer = reconstructed_flat.reshape(layer_shape)
            reconstructed_dnn.append(reconstructed_layer)

        print("Reconstruction complete!")
        return reconstructed_dnn

    def aggregate_filtered_dnns(self, filtered_dnn_list, weights=None, save_path=None):
        num_clients = len(filtered_dnn_list)
        if num_clients == 0:
            raise ValueError("filtered_dnn_list cannot be empty")

        if weights is None:
            weights = [1.0 / num_clients] * num_clients
        else:
            if len(weights) != num_clients:
                raise ValueError("Number of weights must match number of clients")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        equal_weights = np.allclose(
            weights, weights[0], rtol=1e-7, atol=1e-12
        )

        metadata = filtered_dnn_list[0]['metadata'].copy()
        num_layers = int(metadata.get('num_layers', 0))

        totalSize_encrypted = 0
        aggregated_encrypted = None

        if all(fd.get('encrypted_data') is not None for fd in filtered_dnn_list):
            if equal_weights:
                aggregated_encrypted = filtered_dnn_list[0]['encrypted_data']
                for i in range(1, num_clients):
                    aggregated_encrypted = self.ckks.encrypted_add(
                        aggregated_encrypted,
                        filtered_dnn_list[i]['encrypted_data'],
                    )
                aggregated_encrypted = self.ckks.encrypted_multiply_scalar(
                    aggregated_encrypted, weights[0]
                )
            else:
                aggregated_encrypted = self.ckks.encrypted_multiply_scalar(
                    filtered_dnn_list[0]['encrypted_data'], weights[0]
                )
                for i in range(1, num_clients):
                    weighted_enc = self.ckks.encrypted_multiply_scalar(
                        filtered_dnn_list[i]['encrypted_data'], weights[i]
                    )
                    aggregated_encrypted = self.ckks.encrypted_add(
                        aggregated_encrypted, weighted_enc
                    )

            totalSize_encrypted = self.ckks.measure_encrypted_size(
                aggregated_encrypted
            )
        else:
            print("    - Encrypted aggregation skipped (missing encrypted_data for one or more clients)")
            aggregated_encrypted = None

        totalSize_plaintext = 0
        aggregated_plaintext = None

        if all(fd.get('plaintext_data') is not None for fd in filtered_dnn_list):
            if equal_weights:
                aggregated_plaintext = filtered_dnn_list[0]['plaintext_data']
                for i in range(1, num_clients):
                    aggregated_plaintext = self.ckks.plaintext_add(
                        aggregated_plaintext,
                        filtered_dnn_list[i]['plaintext_data'],
                    )
                aggregated_plaintext = self.ckks.plaintext_multiply_scalar(
                    aggregated_plaintext, weights[0]
                )
            else:
                aggregated_plaintext = self.ckks.plaintext_multiply_scalar(
                    filtered_dnn_list[0]['plaintext_data'], weights[0]
                )
                for i in range(1, num_clients):
                    weighted_pt = self.ckks.plaintext_multiply_scalar(
                        filtered_dnn_list[i]['plaintext_data'], weights[i]
                    )
                    aggregated_plaintext = self.ckks.plaintext_add(
                        aggregated_plaintext, weighted_pt
                    )

            totalSize_plaintext = aggregated_plaintext.get('total_bytes', 0)
        else:
            print("    - Plaintext aggregation skipped (missing plaintext_data for one or more clients)")
            aggregated_plaintext = None

        aggregated_encrypted_sparse = []
        aggregated_plaintext_sparse = []
        for layer_idx in range(num_layers):
            aggregated_encrypted_sparse.append({
                'indices': filtered_dnn_list[0].get('encrypted_weights_sparse', self._empty_sparse(num_layers))[
                    layer_idx].get(
                    'indices', np.array([], dtype=np.int64)
                )
            })
            aggregated_plaintext_sparse.append({
                'indices': filtered_dnn_list[0].get('plaintext_weights_sparse', self._empty_sparse(num_layers))[
                    layer_idx].get(
                    'indices', np.array([], dtype=np.int64)
                )
            })

        aggregated_result = {
            'encrypted_data': aggregated_encrypted,
            'plaintext_data': aggregated_plaintext,
            'encrypted_weights_sparse': aggregated_encrypted_sparse,
            'plaintext_weights_sparse': aggregated_plaintext_sparse,
            'metadata': metadata,
            'totalSize_encrypted': totalSize_encrypted,
            'totalSize_plaintext': totalSize_plaintext,
            'totalSize_Sum': totalSize_plaintext + totalSize_encrypted,
            'size_server_cyphertext': totalSize_encrypted,
            'size_server_plaintext': totalSize_plaintext,
        }

        if save_path:
            self.save_filtered_dnn(aggregated_result, save_path)

        return aggregated_result


# ------------------------------ MaskCrypt-specific functions ------------------------------
def build_consensus_mask(proposals):
    assert proposals, "No Mask Has Been Proposed By Clients!"
    if all(proposal is None for proposal in proposals):
        return None
    assert all(proposal is not None for proposal in proposals), \
        "Mask proposals cannot mix full-encryption sentinels with index arrays"

    encrypt_count = len(proposals[0])
    assert all(len(p) == encrypt_count for p in proposals), \
        f"Size Mismatch: Not all proposals are of length {encrypt_count}"

    interleaved = []
    for i in range(encrypt_count):
        for p in proposals:
            interleaved.append(int(p[i]))

    seen = set()
    ordered_unique = []
    for idx in interleaved:
        if idx not in seen:
            seen.add(idx)
            ordered_unique.append(idx)
            if len(ordered_unique) == encrypt_count:
                break

    return np.array(ordered_unique, dtype=np.int32)


def extractStructure(weights_list):
    structure = []
    for w in weights_list:
        # Check if it's a PyTorch tensor before extracting properties
        if isinstance(w, torch.Tensor):
            shape = tuple(w.shape)
            size = w.numel()
        else:
            shape = w.shape
            size = w.size

        structure.append({
            'shape': shape,
            'size': size
        })
    return structure


def indexMask_to_BinaryMask(global_mask, model_structure):
    if global_mask is None:
        maskBool = [
            np.ones(layer['shape'], dtype=np.uint8)
            for layer in model_structure
        ]
        maskBoolNot = [
            np.zeros(layer['shape'], dtype=np.uint8)
            for layer in model_structure
        ]
        return maskBool, maskBoolNot

    maskBool = []
    maskBoolNot = []

    global_mask = np.asarray(global_mask, dtype=np.int64)
    global_mask.sort()

    ptr = 0
    n = len(global_mask)
    offset = 0

    for layer in model_structure:
        size = layer['size']
        shape = layer['shape']

        layer_mask_flat = np.zeros(size, dtype=np.uint8)

        while ptr < n and global_mask[ptr] < offset + size:
            idx = global_mask[ptr] - offset
            if idx >= 0:
                layer_mask_flat[idx] = 1
            ptr += 1

        layer_mask = layer_mask_flat.reshape(shape)
        maskBool.append(layer_mask)
        maskBoolNot.append(1 - layer_mask)

        offset += size

    return maskBool, maskBoolNot


def flattener(data_list):
    if not data_list or len(data_list) == 0:
        return np.array([], dtype=np.float32)

    # Convert PyTorch tensor to numpy if necessary
    first_item = data_list[0]
    if isinstance(first_item, torch.Tensor):
        first_element = first_item.detach().cpu().numpy()
    else:
        first_element = np.asanyarray(first_item)

    detected_dtype = first_element.dtype

    flats = []
    for item in data_list:
        if isinstance(item, torch.Tensor):
            arr = item.detach().cpu().numpy().ravel().astype(detected_dtype)
        else:
            arr = np.asanyarray(item).ravel().astype(detected_dtype)
        flats.append(arr)

    return np.concatenate(flats)


def reconstructor(flat_array, modelStructure):
    reconstructed = []
    start_idx = 0

    for layer_info in modelStructure:
        end_idx = start_idx + layer_info['size']
        chunk = flat_array[start_idx:end_idx]
        reconstructed.append(chunk.reshape(layer_info['shape']))
        start_idx = end_idx

    return reconstructed
