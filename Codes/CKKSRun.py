import os
import tenseal as ts
import numpy as np
import pickle

ckks_config = {'poly_modulus_degree': 8192,
               'coeff_mod_bit_sizes': [60, 30, 60],
               'global_scale': 2 ** 30}# If weights are small

# For deeper operations (federated averaging, multiple additions)
#         poly_modulus_degree = 16384  # Better security and more slots
#         coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 60]  # More depth
#         global_scale = 2 ** 40  # Good for weights in range [-10, 10]
####################################################
class HE_CKKS():
    def __init__(self, createNew=False, batch_size=8192, baseAddr=None):
        self.savedAddr = os.path.abspath("./Temp/CKKS/").replace("\\", "/") + "/"
        if baseAddr is not None:
            self.savedAddr = baseAddr

        os.makedirs(self.savedAddr, exist_ok=True)

        self.batch_size = batch_size

        self.poly_modulus_degree = ckks_config["poly_modulus_degree"]
        self.coeff_mod_bit_sizes = ckks_config["coeff_mod_bit_sizes"]
        self.global_scale = ckks_config["global_scale"]

        if createNew:
            print("Create New CKKS with TenSEAL")
            # Create TenSEAL context
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            )

            context.global_scale = self.global_scale
            # Galois keys are only needed for slot rotations. MaskCrypt uses
            # slot-wise addition and plaintext multiplication, so generating
            # them would add context setup and serialization overhead.
            # context.generate_galois_keys()

            # Save context
            with open(self.savedAddr + "context.pkl", 'wb') as f:
                f.write(context.serialize(save_secret_key=True))

            self.context = context
        else:
            # Load existing context
            with open(self.savedAddr + "context.pkl", 'rb') as f:
                context_bytes = f.read()
            self.context = ts.context_from(context_bytes)

        self.public_context = None



    @staticmethod
    def _serialized_ciphertexts_size(ciphertexts):
        """Return the actual serialized size of a ciphertext sequence."""
        return sum(len(ciphertext.serialize()) for ciphertext in ciphertexts)

    def measure_encrypted_size(self, encrypted_dict):
        """Measure and cache size only when a ciphertext payload is reported."""
        self._validate_encrypted_structure(encrypted_dict)
        total_bytes = self._serialized_ciphertexts_size(encrypted_dict['ciphertexts'])
        encrypted_dict['total_bytes'] = total_bytes
        return total_bytes

    @staticmethod
    def _shape_signature(shapes):
        return tuple(tuple(int(dimension) for dimension in shape) for shape in shapes)

    def _validate_encrypted_structure(self, encrypted_dict, name="encrypted data"):
        """Validate metadata that defines how logical values map to CKKS slots."""
        required = (
            'ciphertexts', 'shapes', 'layer_sizes', 'n_values', 'n_chunks',
            'total_size', 'packing_mode'
        )
        missing = [key for key in required if key not in encrypted_dict]
        if missing:
            raise ValueError(f"{name} is missing required metadata: {missing}")

        ciphertexts = encrypted_dict['ciphertexts']
        n_chunks = int(encrypted_dict['n_chunks'])
        n_values = int(encrypted_dict['n_values'])
        total_size = int(encrypted_dict['total_size'])
        layer_sizes = tuple(int(size) for size in encrypted_dict['layer_sizes'])
        shapes = self._shape_signature(encrypted_dict['shapes'])
        max_slots = self.poly_modulus_degree // 2
        expected_chunks = int(np.ceil(n_values / max_slots))

        if n_values < 0 or total_size < 0:
            raise ValueError(f"{name} has negative size metadata")
        if len(ciphertexts) != n_chunks or n_chunks != expected_chunks:
            raise ValueError(
                f"{name} has inconsistent chunk metadata: "
                f"ciphertexts={len(ciphertexts)}, n_chunks={n_chunks}, "
                f"expected={expected_chunks}"
            )
        if sum(layer_sizes) != total_size:
            raise ValueError(f"{name} layer sizes do not sum to total_size")
        if len(shapes) != len(layer_sizes) or any(
                int(np.prod(shape)) != size for shape, size in zip(shapes, layer_sizes)):
            raise ValueError(f"{name} shapes and layer sizes are inconsistent")

        packing_mode = encrypted_dict['packing_mode']
        if packing_mode != 'dense':
            raise ValueError(f"{name} uses unsupported packing_mode {packing_mode!r}")
        if n_values != total_size:
            raise ValueError(
                f"{name} declares dense packing but does not contain every logical position"
            )

        for chunk_index, ciphertext in enumerate(ciphertexts):
            if ciphertext.size() != max_slots:
                raise ValueError(
                    f"{name} ciphertext chunk {chunk_index} has {ciphertext.size()} slots; "
                    f"expected {max_slots}"
                )

        return {
            'shapes': shapes,
            'layer_sizes': layer_sizes,
            'n_values': n_values,
            'n_chunks': n_chunks,
            'total_size': total_size,
            'packing_mode': packing_mode,
        }

    def _flatten_model(self, model):
        """Flatten entire model (list of layers) into single array"""
        flat_arrays = []
        shapes = []
        layer_sizes = []

        for layer in model:
            layer_array = np.array(layer).astype(np.float64)
            shapes.append(layer_array.shape)
            layer_sizes.append(layer_array.size)
            flat_arrays.append(layer_array.flatten())

        # Concatenate all layers into single flat array
        full_flat = np.concatenate(flat_arrays)

        return full_flat, shapes, layer_sizes

    def _unflatten_model(self, flat_array, shapes, layer_sizes):
        """Reconstruct model from flat array"""
        model = []
        offset = 0

        for shape, size in zip(shapes, layer_sizes):
            layer_flat = flat_array[offset:offset + size]
            layer = layer_flat.reshape(shape)
            model.append(layer)
            offset += size

        return model

    def encrypt_data(self, data, mask=None):
        """
        Encrypt DNN model or mask.

        Args:
            data: DNN model (list of numpy arrays) or mask (list of numpy arrays)
            mask: Optional mask (list of numpy arrays). If provided, encrypt
                  the dense element-wise product data*mask.
        Returns:
            Dictionary containing encrypted data and metadata
        """
        # Flatten the entire model/mask
        if mask is None:
            flat_data, shapes, layer_sizes = self._flatten_model(data)
            values_to_encrypt = flat_data

        else:
            # Encrypt the dense data*mask product.
            flat_data, data_shapes, data_layer_sizes = self._flatten_model(data)
            flat_mask, mask_shapes, mask_layer_sizes = self._flatten_model(mask)

            # Ensure data and mask have same structure
            assert flat_data.shape == flat_mask.shape, "Data and mask must have same shape"

            values_to_encrypt = flat_data * flat_mask

            shapes = data_shapes
            layer_sizes = data_layer_sizes

        # Encrypt values in chunks
        max_slots = self.poly_modulus_degree // 2
        n_values = len(values_to_encrypt)
        n_chunks = int(np.ceil(n_values / max_slots))

        ciphertexts = []
        for i in range(n_chunks):
            start_idx = i * max_slots
            end_idx = min((i + 1) * max_slots, n_values)
            chunk = values_to_encrypt[start_idx:end_idx]

            # Pad to max_slots
            if chunk.size < max_slots:
                chunk = np.pad(chunk, (0, max_slots - chunk.size))

            ciphertexts.append(ts.ckks_vector(self.context, chunk.tolist()))

        total_bytes = self._serialized_ciphertexts_size(ciphertexts)

        # Return encrypted data with metadata
        return {
            'ciphertexts': ciphertexts,
            'shapes': shapes,
            'layer_sizes': layer_sizes,
            'n_values': n_values,
            'n_chunks': n_chunks,
            'total_size': sum(layer_sizes),
            'total_bytes': total_bytes,
            'packing_mode': 'dense'
        }

    def decrypt_data(self, encrypted_dict):
        """
        Decrypt data back to original model structure.

        Args:
            encrypted_dict: Dictionary containing encrypted data and metadata

        Returns:
            Decrypted model (list of numpy arrays)
        """
        layout = self._validate_encrypted_structure(encrypted_dict)
        ciphertexts = encrypted_dict['ciphertexts']
        shapes = encrypted_dict['shapes']
        layer_sizes = encrypted_dict['layer_sizes']
        n_values = encrypted_dict['n_values']
        total_size = layout['total_size']

        # Decrypt all chunks
        max_slots = self.poly_modulus_degree // 2
        decrypted_chunks = []

        for ciphertext in ciphertexts:
            chunk = np.array(ciphertext.decrypt())
            decrypted_chunks.append(chunk)

        # Concatenate and trim to actual number of values
        parts = []
        remaining = n_values
        for chunk in decrypted_chunks:
            take = min(max_slots, remaining)
            if take <= 0:
                break
            parts.append(chunk[:take])
            remaining -= take

        decrypted_values = (
            np.concatenate(parts)[:n_values]
            if parts else np.empty(0, dtype=np.float64)
        )
        full_flat = np.asarray(decrypted_values, dtype=np.float64)

        # Unflatten back to model structure
        model = self._unflatten_model(full_flat, shapes, layer_sizes)

        return model

    def save_encrypted_data(self, encrypted_dict, save_path):
        """
        Save encrypted data to disk.

        Args:
            encrypted_dict: Dictionary containing encrypted data
            save_path: Path to save encrypted data
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        layout = self._validate_encrypted_structure(encrypted_dict)

        # Serialize each ciphertext exactly once: the same bytes are written
        # and counted instead of performing a separate size-only pass.
        cipher_dir = f"{save_path}_ciphertexts"
        os.makedirs(cipher_dir, exist_ok=True)
        total_bytes = 0
        for i, ciphertext in enumerate(encrypted_dict['ciphertexts']):
            ciphertext_bytes = ciphertext.serialize()
            total_bytes += len(ciphertext_bytes)
            with open(f"{cipher_dir}/chunk_{i}.bin", 'wb') as f:
                f.write(ciphertext_bytes)
        encrypted_dict['total_bytes'] = total_bytes

        metadata = {
            'shapes': [s if isinstance(s, list) else s for s in encrypted_dict['shapes']],
            'layer_sizes': encrypted_dict['layer_sizes'],
            'n_values': encrypted_dict['n_values'],
            'n_chunks': encrypted_dict['n_chunks'],
            'total_size': encrypted_dict['total_size'],
            'total_bytes': total_bytes,
            'packing_mode': layout['packing_mode'],
        }

        with open(f"{save_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

    def load_encrypted_data(self, save_path):
        """
        Load encrypted data from disk.

        Args:
            save_path: Path where encrypted data was saved

        Returns:
            Dictionary containing encrypted data
        """
        # Load metadata
        with open(f"{save_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Load ciphertexts
        cipher_dir = f"{save_path}_ciphertexts"
        ciphertexts = []
        total_bytes = 0

        for i in range(metadata['n_chunks']):
            with open(f"{cipher_dir}/chunk_{i}.bin", 'rb') as f:
                ct_bytes = f.read()
                total_bytes += len(ct_bytes)
                ciphertext = ts.ckks_vector_from(self.context, ct_bytes)
                ciphertexts.append(ciphertext)

        total_size = int(metadata['total_size'])
        packing_mode = metadata['packing_mode']

        loaded = {
            'ciphertexts': ciphertexts,
            'shapes': metadata['shapes'],
            'layer_sizes': metadata['layer_sizes'],
            'n_values': metadata['n_values'],
            'n_chunks': metadata['n_chunks'],
            'total_size': total_size,
            'total_bytes': total_bytes,
            'packing_mode': packing_mode,
        }
        self._validate_encrypted_structure(loaded, "loaded encrypted data")
        return loaded

    def encrypted_add(self, encrypted_dict1, encrypted_dict2):
        """
        Add two encrypted data (homomorphic addition).
        Both operands must have the same dense layout.

        Args:
            encrypted_dict1: First encrypted data
            encrypted_dict2: Second encrypted data

        Returns:
            Result of addition
        """
        layout1 = self._validate_encrypted_structure(encrypted_dict1, "first encrypted operand")
        layout2 = self._validate_encrypted_structure(encrypted_dict2, "second encrypted operand")
        comparable_fields = (
            'shapes', 'layer_sizes', 'n_values', 'n_chunks', 'total_size',
            'packing_mode'
        )
        mismatches = [
            field for field in comparable_fields if layout1[field] != layout2[field]
        ]
        if mismatches:
            raise ValueError(
                "Encrypted data layouts are incompatible for slot-wise addition; "
                f"mismatched metadata: {', '.join(mismatches)}"
            )

        result_ciphertexts = []
        for ct1, ct2 in zip(encrypted_dict1['ciphertexts'], encrypted_dict2['ciphertexts']):
            result_ciphertexts.append(ct1 + ct2)

        return {
            'ciphertexts': result_ciphertexts,
            'shapes': encrypted_dict1['shapes'],
            'layer_sizes': encrypted_dict1['layer_sizes'],
            'n_values': encrypted_dict1['n_values'],
            'n_chunks': encrypted_dict1['n_chunks'],
            'total_size': encrypted_dict1['total_size'],
            'packing_mode': 'dense',
        }

    def encrypted_multiply_scalar(self, encrypted_dict, scalar):
        """
        Multiply encrypted data by scalar (homomorphic multiplication).

        Args:
            encrypted_dict: Encrypted data
            scalar: Scalar value

        Returns:
            Result of multiplication
        """
        layout = self._validate_encrypted_structure(encrypted_dict)
        result_ciphertexts = []

        # Get the slot count for creating scalar vector
        max_slots = self.poly_modulus_degree // 2

        # Create a vector filled with the scalar value
        scalar_vector = [float(scalar)] * max_slots

        for ct in encrypted_dict['ciphertexts']:
            # Multiply ciphertext by scalar vector
            result_ciphertexts.append(ct * scalar_vector)

        return {
            'ciphertexts': result_ciphertexts,
            'shapes': encrypted_dict['shapes'],
            'layer_sizes': encrypted_dict['layer_sizes'],
            'n_values': encrypted_dict['n_values'],
            'n_chunks': encrypted_dict['n_chunks'],
            'total_size': encrypted_dict['total_size'],
            'packing_mode': 'dense',
        }


    def encode_data(self, data, mask=None):
        """
        Pack DNN model or mask values as a raw float32 plaintext payload.

        Args:
            data: DNN model (list of numpy arrays) or mask (list of numpy arrays)
            mask: Optional mask (list of numpy arrays). If provided, encode
                  the dense element-wise product data*mask.
        Returns:
            Dictionary containing a contiguous raw values array and metadata.
        """
        # Flatten the entire model/mask
        if mask is None:
            flat_data, shapes, layer_sizes = self._flatten_model(data)
            values_to_encode = flat_data

        else:
            # Encode the dense data*mask product.
            flat_data, data_shapes, data_layer_sizes = self._flatten_model(data)
            flat_mask, mask_shapes, mask_layer_sizes = self._flatten_model(mask)

            if flat_data.shape != flat_mask.shape:
                raise ValueError("Data and mask must have same shape")

            values_to_encode = flat_data * flat_mask

            shapes = data_shapes
            layer_sizes = data_layer_sizes

        # Raw values are not padded; n_chunks records their deterministic layout.
        max_slots = self.poly_modulus_degree // 2
        n_values = len(values_to_encode)
        n_chunks = int(np.ceil(n_values / max_slots))
        values = np.ascontiguousarray(values_to_encode, dtype=np.float32)

        return {
            'values': values,
            'shapes': shapes,
            'layer_sizes': layer_sizes,
            'n_values': n_values,
            'n_chunks': n_chunks,
            'total_size': sum(layer_sizes),
            'total_bytes': values.nbytes,
            'packing_mode': 'dense',
            'serialization_format': 'raw_float32',
            'dtype': values.dtype.str,
        }

    def _extract_plaintext_values(self, plaintext_dict):
        """Return contiguous float32 values from a dense plaintext payload."""
        n_values = int(plaintext_dict['n_values'])
        values = np.asarray(plaintext_dict['values'], dtype=np.float32).reshape(-1)

        if values.size != n_values:
            raise ValueError(
                f"Plaintext payload contains {values.size} values; expected {n_values}"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _validate_plaintext_structure(self, plaintext_dict, name="plaintext data"):
        required = (
            'values', 'shapes', 'layer_sizes', 'n_values', 'n_chunks',
            'total_size', 'packing_mode'
        )
        missing = [key for key in required if key not in plaintext_dict]
        if missing:
            raise ValueError(f"{name} is missing required metadata: {missing}")

        values = self._extract_plaintext_values(plaintext_dict)
        shapes = self._shape_signature(plaintext_dict['shapes'])
        layer_sizes = tuple(int(size) for size in plaintext_dict['layer_sizes'])
        n_values = int(plaintext_dict['n_values'])
        n_chunks = int(plaintext_dict['n_chunks'])
        total_size = int(plaintext_dict['total_size'])
        expected_chunks = int(np.ceil(n_values / (self.poly_modulus_degree // 2)))

        if n_values < 0 or total_size < 0:
            raise ValueError(f"{name} has negative size metadata")
        if n_chunks != expected_chunks:
            raise ValueError(
                f"{name} has n_chunks={n_chunks}; expected {expected_chunks}"
            )
        if sum(layer_sizes) != total_size:
            raise ValueError(f"{name} layer sizes do not sum to total_size")
        if len(shapes) != len(layer_sizes) or any(
                int(np.prod(shape)) != size for shape, size in zip(shapes, layer_sizes)):
            raise ValueError(f"{name} shapes and layer sizes are inconsistent")

        packing_mode = plaintext_dict['packing_mode']
        if packing_mode != 'dense':
            raise ValueError(f"{name} uses unsupported packing_mode {packing_mode!r}")
        if n_values != total_size:
            raise ValueError(
                f"{name} declares dense packing but does not contain every logical position"
            )

        return {
            'values': values,
            'shapes': shapes,
            'layer_sizes': layer_sizes,
            'n_values': n_values,
            'n_chunks': n_chunks,
            'total_size': total_size,
            'packing_mode': packing_mode,
        }

    @staticmethod
    def _raw_plaintext_result(values, template):
        values = np.ascontiguousarray(values, dtype=np.float32)
        return {
            'values': values,
            'shapes': template['shapes'],
            'layer_sizes': template['layer_sizes'],
            'n_values': int(template['n_values']),
            'n_chunks': int(template['n_chunks']),
            'total_size': int(template['total_size']),
            'total_bytes': values.nbytes,
            'packing_mode': 'dense',
            'serialization_format': 'raw_float32',
            'dtype': values.dtype.str,
        }

    def decode_data(self, plaintext_dict):
        """
        Decode plaintext data back to original model structure.

        Args:
            plaintext_dict: Dictionary containing plaintext data and metadata

        Returns:
            Decoded model (list of numpy arrays)
        """
        layout = self._validate_plaintext_structure(plaintext_dict)

        full_flat = layout['values'].copy()

        # Unflatten back to model structure
        model = self._unflatten_model(
            full_flat, plaintext_dict['shapes'], plaintext_dict['layer_sizes']
        )

        return model

    def plaintext_add(self, plaintext_dict1, plaintext_dict2):
        """
        Add two plaintext data (element-wise addition).
        Both operands must have the same dense layout.

        Args:
            plaintext_dict1: First plaintext data
            plaintext_dict2: Second plaintext data

        Returns:
            Result of addition
        """
        layout1 = self._validate_plaintext_structure(plaintext_dict1, "first plaintext operand")
        layout2 = self._validate_plaintext_structure(plaintext_dict2, "second plaintext operand")
        comparable_fields = (
            'shapes', 'layer_sizes', 'n_values', 'n_chunks', 'total_size',
            'packing_mode'
        )
        mismatches = [
            field for field in comparable_fields if layout1[field] != layout2[field]
        ]
        if mismatches:
            raise ValueError(
                "Plaintext data layouts are incompatible for addition; "
                f"mismatched metadata: {', '.join(mismatches)}"
            )

        return self._raw_plaintext_result(
            layout1['values'] + layout2['values'], plaintext_dict1
        )

    def plaintext_multiply_scalar(self, plaintext_dict, scalar):
        """
        Multiply plaintext data by scalar (element-wise multiplication).

        Args:
            plaintext_dict: Plaintext data
            scalar: Scalar value

        Returns:
            Result of multiplication
        """
        layout = self._validate_plaintext_structure(plaintext_dict)
        result_values = np.multiply(
            layout['values'], np.float32(scalar), dtype=np.float32
        )
        return self._raw_plaintext_result(
            result_values, plaintext_dict
        )


    def save_plaintext_data(self, plaintext_dict, save_path):
        """
        Save plaintext data to disk.

        Args:
            plaintext_dict: Dictionary containing plaintext data
            save_path: Path to save plaintext data
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        layout = self._validate_plaintext_structure(plaintext_dict)
        values = layout['values']
        raw_bytes = values.tobytes(order='C')

        metadata = {
            'shapes': [s if isinstance(s, list) else s for s in plaintext_dict['shapes']],
            'layer_sizes': plaintext_dict['layer_sizes'],
            'n_values': plaintext_dict['n_values'],
            'n_chunks': plaintext_dict['n_chunks'],
            'total_size': plaintext_dict['total_size'],
            'total_bytes': len(raw_bytes),
            'packing_mode': layout['packing_mode'],
            'serialization_format': 'raw_float32',
            'dtype': values.dtype.str,
        }

        with open(f"{save_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        with open(f"{save_path}_values.bin", 'wb') as f:
            f.write(raw_bytes)

    def load_plaintext_data(self, save_path):
        """
        Load plaintext data from disk.

        Args:
            save_path: Path where plaintext data was saved

        Returns:
            Dictionary containing plaintext data
        """
        # Load metadata
        with open(f"{save_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        n_values = int(metadata['n_values'])
        if metadata['serialization_format'] != 'raw_float32':
            raise ValueError("Unsupported plaintext serialization format")
        with open(f"{save_path}_values.bin", 'rb') as f:
            raw_bytes = f.read()
        dtype = np.dtype(metadata['dtype'])
        values = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32, copy=True)

        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.size != n_values:
            raise ValueError(
                f"Saved plaintext contains {values.size} values; expected {n_values}"
            )
        total_size = int(metadata['total_size'])
        packing_mode = metadata['packing_mode']

        loaded = {
            'values': values,
            'shapes': metadata['shapes'],
            'layer_sizes': metadata['layer_sizes'],
            'n_values': n_values,
            'n_chunks': metadata['n_chunks'],
            'total_size': total_size,
            'total_bytes': values.nbytes,
            'packing_mode': packing_mode,
            'serialization_format': 'raw_float32',
            'dtype': values.dtype.str,
        }
        self._validate_plaintext_structure(loaded, "loaded plaintext data")
        return loaded
