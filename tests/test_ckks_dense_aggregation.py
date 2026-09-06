import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Codes.CKKSRun import HE_CKKS
from Codes.functions_mainAlg import FilteredDNNEncryption, build_mask_plan


class DenseCKKSAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._context_directory = tempfile.TemporaryDirectory()
        context_path = cls._context_directory.name + os.sep
        cls.ckks = HE_CKKS(createNew=True, baseAddr=context_path)

    @classmethod
    def tearDownClass(cls):
        cls._context_directory.cleanup()

    def assert_decrypts_to(self, encrypted, expected, atol=1e-3):
        decrypted = self.ckks.decrypt_data(encrypted)
        np.testing.assert_allclose(decrypted[0], expected, rtol=0.0, atol=atol)

    def test_dense_vector_with_internal_zero_round_trips_without_sparse_metadata(self):
        values = np.array([8.0, 0.0, 4.0])

        encrypted = self.ckks.encrypt_data([values])

        self.assertEqual(encrypted['packing_mode'], 'dense')
        self.assertEqual(encrypted['n_values'], 3)
        self.assertNotIn('non_zero_indices', encrypted)
        self.assertNotIn('filter_zeros', encrypted)
        self.assert_decrypts_to(encrypted, values)

    def test_clients_with_different_zero_patterns_aggregate_by_consensus_slot(self):
        class Config:
            pass

        values_a = [np.array([8.0, 0.0, 4.0])]
        values_b = [np.array([3.0, 7.0, 0.0])]
        mask = [np.ones(3)]
        mask_not = [np.zeros(3)]
        mask_plan = build_mask_plan(mask, mask_not)

        with tempfile.TemporaryDirectory() as directory:
            handler = FilteredDNNEncryption(Config(), self.ckks, directory)
            with patch('Codes.functions_mainAlg.excelHelper.update'):
                client_a, _ = handler.filteringDNN(values_a, mask_plan)
                client_b, _ = handler.filteringDNN(values_b, mask_plan)
            self.assertIs(
                client_a['encrypted_weights_sparse'],
                client_b['encrypted_weights_sparse'],
            )
            self.assertIs(
                client_a['encrypted_weights_sparse'],
                mask_plan['encrypted_weights_sparse'],
            )
            with patch.object(
                    self.ckks,
                    'encrypted_multiply_scalar',
                    wraps=self.ckks.encrypted_multiply_scalar,
            ) as multiply_scalar:
                filtered_aggregate = handler.aggregate_filtered_dnns(
                    [client_a, client_b], weights=[0.5, 0.5]
                )
            self.assertEqual(multiply_scalar.call_count, 1)
            reconstructed = handler.reconstructDNN(filtered_aggregate)

        aggregate = filtered_aggregate['encrypted_data']
        expected = np.array([5.5, 3.5, 2.0])
        self.assertNotIn('non_zero_indices', aggregate)
        self.assert_decrypts_to(aggregate, expected)
        np.testing.assert_allclose(reconstructed[0], expected, rtol=0.0, atol=1e-3)
        self.assertEqual(
            aggregate['total_bytes'],
            sum(len(ciphertext.serialize()) for ciphertext in aggregate['ciphertexts'])
        )
        self.assertEqual(
            filtered_aggregate['size_server_cyphertext'], aggregate['total_bytes']
        )
        self.assertEqual(filtered_aggregate['size_server_plaintext'], 0)

    def test_unequal_client_weights_keep_per_client_scaling(self):
        class Config:
            pass

        mask_plan = build_mask_plan([np.ones(2)], [np.zeros(2)])
        with tempfile.TemporaryDirectory() as directory:
            handler = FilteredDNNEncryption(Config(), self.ckks, directory)
            with patch('Codes.functions_mainAlg.excelHelper.update'):
                client_a, _ = handler.filteringDNN(
                    [np.array([4.0, 8.0])], mask_plan
                )
                client_b, _ = handler.filteringDNN(
                    [np.array([12.0, 16.0])], mask_plan
                )
            with patch.object(
                    self.ckks,
                    'encrypted_multiply_scalar',
                    wraps=self.ckks.encrypted_multiply_scalar,
            ) as multiply_scalar:
                aggregate = handler.aggregate_filtered_dnns(
                    [client_a, client_b], weights=[0.25, 0.75]
                )

        self.assertEqual(multiply_scalar.call_count, 2)
        self.assert_decrypts_to(
            aggregate['encrypted_data'], np.array([10.0, 14.0])
        )

    def test_intermediate_ckks_arithmetic_does_not_serialize_for_size(self):
        encrypted = self.ckks.encrypt_data([np.array([1.0, 2.0])])

        added = self.ckks.encrypted_add(encrypted, encrypted)
        scaled = self.ckks.encrypted_multiply_scalar(added, 0.5)

        self.assertNotIn('total_bytes', added)
        self.assertNotIn('total_bytes', scaled)
        self.assert_decrypts_to(scaled, np.array([1.0, 2.0]))

    def test_save_load_preserves_dense_layout_and_actual_size(self):
        encrypted = self.ckks.encrypt_data([np.array([2.0, 0.0, -1.0])])

        with tempfile.TemporaryDirectory() as directory:
            save_path = str(Path(directory, 'dense'))
            self.ckks.save_encrypted_data(encrypted, save_path)
            with open(save_path + '_metadata.pkl', 'rb') as metadata_file:
                saved_metadata = pickle.load(metadata_file)
            loaded = self.ckks.load_encrypted_data(save_path)

        self.assertEqual(saved_metadata['packing_mode'], 'dense')
        self.assertNotIn('non_zero_indices', saved_metadata)
        self.assertNotIn('filter_zeros', saved_metadata)
        self.assertEqual(loaded['packing_mode'], 'dense')
        self.assertNotIn('non_zero_indices', loaded)
        self.assertNotIn('filter_zeros', loaded)
        self.assertEqual(
            loaded['total_bytes'],
            sum(len(ciphertext.serialize()) for ciphertext in loaded['ciphertexts'])
        )
        self.assert_decrypts_to(loaded, np.array([2.0, 0.0, -1.0]))

    def test_filtering_keeps_selected_zero_and_reconstructs_original_positions(self):
        class Config:
            pass

        weights = [
            np.array([8.0, 0.0, 4.0, 9.0]),
            np.array([0.0, -2.0]),
        ]
        mask = [
            np.array([1.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        mask_not = [1.0 - layer_mask for layer_mask in mask]
        mask_plan = build_mask_plan(mask, mask_not)

        with tempfile.TemporaryDirectory() as directory:
            handler = FilteredDNNEncryption(Config(), self.ckks, directory)
            with patch('Codes.functions_mainAlg.excelHelper.update'):
                filtered, _ = handler.filteringDNN(weights, mask_plan)
            reconstructed = handler.reconstructDNN(filtered)

        encrypted = filtered['encrypted_data']
        self.assertNotIn('originalDnn', filtered)
        for sparse_layers in (
                filtered['encrypted_weights_sparse'],
                filtered['plaintext_weights_sparse']):
            self.assertTrue(all('values' not in layer for layer in sparse_layers))
        self.assertEqual(encrypted['packing_mode'], 'dense')
        self.assertEqual(encrypted['n_values'], 4)
        self.assertNotIn('non_zero_indices', encrypted)
        self.assertEqual(
            filtered['size_cyphertext'],
            sum(len(ciphertext.serialize()) for ciphertext in encrypted['ciphertexts'])
        )
        self.assertEqual(
            filtered['size_plaintext'],
            filtered['plaintext_data']['values'].nbytes
        )
        self.assertEqual(filtered['plaintext_data']['values'].dtype, np.float32)
        self.assert_decrypts_to(encrypted, np.array([8.0, 0.0, 4.0, -2.0]))
        for actual_layer, expected_layer in zip(reconstructed, weights):
            np.testing.assert_allclose(actual_layer, expected_layer, rtol=0.0, atol=1e-3)

    def test_add_rejects_same_chunk_count_with_incompatible_layout(self):
        three_values = self.ckks.encrypt_data([np.array([1.0, 2.0, 3.0])])
        four_values = self.ckks.encrypt_data([np.array([1.0, 2.0, 3.0, 4.0])])
        self.assertEqual(three_values['n_chunks'], four_values['n_chunks'])

        with self.assertRaisesRegex(ValueError, 'incompatible'):
            self.ckks.encrypted_add(three_values, four_values)

    def test_raw_plaintext_aggregation_and_save_load(self):
        client_a = self.ckks.encode_data([np.array([8.0, 0.0, 4.0])])
        client_b = self.ckks.encode_data([np.array([3.0, 7.0, 0.0])])

        self.assertNotIn('plaintexts', client_a)
        self.assertEqual(client_a['serialization_format'], 'raw_float32')
        self.assertEqual(client_a['total_bytes'], 3 * np.dtype(np.float32).itemsize)
        weighted_a = self.ckks.plaintext_multiply_scalar(client_a, 0.5)
        weighted_b = self.ckks.plaintext_multiply_scalar(client_b, 0.5)
        aggregate = self.ckks.plaintext_add(weighted_a, weighted_b)
        np.testing.assert_allclose(
            self.ckks.decode_data(aggregate)[0],
            np.array([5.5, 3.5, 2.0], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )

        with tempfile.TemporaryDirectory() as directory:
            save_path = str(Path(directory, 'raw_plaintext'))
            self.ckks.save_plaintext_data(aggregate, save_path)
            self.assertEqual(Path(save_path + '_values.bin').stat().st_size, 12)
            loaded = self.ckks.load_plaintext_data(save_path)

        self.assertEqual(loaded['serialization_format'], 'raw_float32')
        self.assertEqual(loaded['total_bytes'], 12)
        np.testing.assert_array_equal(loaded['values'], aggregate['values'])

if __name__ == '__main__':
    unittest.main()
