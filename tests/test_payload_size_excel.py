import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from Codes import excelHelper


class PayloadSizeExcelTests(unittest.TestCase):
    def test_client_and_server_payload_sizes_are_combined_by_round(self):
        class Config:
            excelAddr = None
            currentEdge = 0
            num_clients = 2
            rounds = 1

        cfg = Config()

        with tempfile.TemporaryDirectory() as directory:
            cfg.excelAddr = str(Path(directory, 'round_0'))
            excelHelper.create(cfg)

            for client_index, plaintext_size, cyphertext_size in (
                    (0, 101, 1001), (1, 202, 2002)):
                cfg.currentEdge = client_index
                excelHelper.update(cfg, {
                    'size_plaintext': plaintext_size,
                    'size_cyphertext': cyphertext_size,
                })

            excelHelper.update(cfg, {
                'size_server_plaintext': 303,
                'size_server_cyphertext': 3003,
                'size_server_sensitivity_map': 4004,
                'time_ClientMaskProposal': 1.25,
                'time_sensitivityMapsAggregation': 0.75,
                'time_MaskDecryption': 0.5,
                'time_MaskGen': 2.5,
            })

            metrics_path = Path(directory, 'round_0.jsonl')
            self.assertEqual(metrics_path.read_text(encoding='utf-8'), '')
            self.assertFalse(Path(directory, 'round_0.xlsx').exists())
            excelHelper.flush(cfg)
            self.assertEqual(
                len(metrics_path.read_text(encoding='utf-8').splitlines()), 3
            )

            combined_path = str(Path(directory, 'combined'))
            sheets = [
                'size_plaintext', 'size_cyphertext', 'size_sensitivity_map',
                'Times',
            ]
            excelHelper.combineExcels(
                cfg,
                baseAddr=str(Path(directory, 'round_%d')),
                targetSaveAddr=combined_path,
                sheetList=sheets,
            )

            workbook = load_workbook(combined_path + '.xlsx', read_only=True)
            try:
                self.assertEqual(workbook['size_plaintext']['A1'].value, 'Client 1')
                self.assertEqual(workbook['size_plaintext']['B1'].value, 'Client 2')
                self.assertEqual(workbook['size_plaintext']['A2'].value, 101)
                self.assertEqual(workbook['size_plaintext']['B2'].value, 202)
                self.assertEqual(workbook['size_plaintext']['C1'].value, 'Server')
                self.assertEqual(workbook['size_plaintext']['C2'].value, 303)
                self.assertEqual(workbook['size_cyphertext']['A2'].value, 1001)
                self.assertEqual(workbook['size_cyphertext']['B2'].value, 2002)
                self.assertEqual(workbook['size_cyphertext']['C1'].value, 'Server')
                self.assertEqual(workbook['size_cyphertext']['C2'].value, 3003)
                self.assertEqual(
                    workbook['size_sensitivity_map']['C1'].value, 'Server'
                )
                self.assertEqual(
                    workbook['size_sensitivity_map']['C2'].value, 4004
                )
                self.assertEqual(workbook['Times']['J1'].value, 'ClientMaskProposal')
                self.assertEqual(
                    workbook['Times']['K1'].value,
                    'sensitivityMapsAggregation',
                )
                self.assertEqual(workbook['Times']['L1'].value, 'MaskDecryption')
                self.assertEqual(workbook['Times']['M1'].value, 'MaskGen')
                self.assertNotIn(
                    'getSensitivity',
                    [cell.value for cell in workbook['Times'][1]],
                )
                self.assertEqual(workbook['Times']['J2'].value, 1.25)
                self.assertEqual(workbook['Times']['K2'].value, 0.75)
                self.assertEqual(workbook['Times']['L2'].value, 0.5)
                self.assertEqual(workbook['Times']['M2'].value, 2.5)
            finally:
                workbook.close()


if __name__ == '__main__':
    unittest.main()
