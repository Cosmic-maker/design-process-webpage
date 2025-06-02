import unittest
import pandas as pd
import numpy as np

# Dummy-Funktionen, die du in deinen echten Code einbauen kannst
def perform_correspondence_analysis(processes):
    return {proc: f"Analyse von {proc}" for proc in processes}

def perform_cumulative_occurrence_analysis(processes):
    return [f"Diagramm für {proc}" for proc in processes]

def perform_markov_chain_analysis(data):
    codes = ["F", "B", "S"]
    mat = pd.DataFrame(np.array([
        [0.1, 0.7, 0.2],
        [0.4, 0.4, 0.2],
        [0.3, 0.3, 0.4],
    ]), index=codes, columns=codes)
    return mat


class Test(unittest.TestCase):

    def test_correspondence_analysis_selection(self):
        processes = ["Designprozess1", "Designprozess2", "Designprozess3"]
        result = perform_correspondence_analysis(processes)
        self.assertEqual(len(result), len(processes))
        for proc in processes:
            self.assertIn(proc, result)
            self.assertIsInstance(result[proc], str)

    def test_cumulative_occurrence_analysis_diagrams(self):
        processes = ["Designprozess1", "Designprozess2"]
        diagrams = perform_cumulative_occurrence_analysis(processes)
        self.assertEqual(len(diagrams), len(processes))
        for diagram in diagrams:
            self.assertIsInstance(diagram, str)

    def test_markov_chain_analysis_transition_matrix(self):
        data = "dummy_data"
        matrix = perform_markov_chain_analysis(data)
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        # Prüfe, ob alle Werte zwischen 0 und 1 liegen
        self.assertTrue(((matrix.values >= 0) & (matrix.values <= 1)).all())


if __name__ == "__main__":
    unittest.main()
