import unittest
import pandas as pd
import numpy as np
import os

# Dummy-Funktionen, die du in deinen echten Code einbauen kannst
def perform_correspondence_analysis(processes):
    # Simuliere das Erzeugen von Dateien
    filenames = []
    for proc in processes:
        filename = f"{proc}_correspondence.png"
        with open(filename, "w") as f:
            f.write("dummy diagram content")
        filenames.append(filename)
    return filenames

def perform_cumulative_occurrence_analysis(processes):
    filenames = []
    for proc in processes:
        filename = f"{proc}_cumulative.png"
        with open(filename, "w") as f:
            f.write("dummy diagram content")
        filenames.append(filename)
    return filenames

def perform_markov_chain_analysis(data):
    codes = ["F", "B", "S"]
    mat = pd.DataFrame(np.array([
        [0.1, 0.7, 0.2],
        [0.4, 0.4, 0.2],
        [0.3, 0.3, 0.4],
    ]), index=codes, columns=codes)
    return mat


class Test(unittest.TestCase):

    def setUp(self):
        self.generated_files = []

    def tearDown(self):
        # Lösche alle Dateien, die während der Tests erstellt wurden
        for file in self.generated_files:
            if os.path.exists(file):
                os.remove(file)

    def test_correspondence_analysis_selection(self):
        processes = ["Designprozess1", "Designprozess2", "Designprozess3"]
        result = perform_correspondence_analysis(processes)
        self.assertEqual(len(result), len(processes))
        for file in result:
            self.assertTrue(os.path.exists(file))
            self.generated_files.append(file)

    def test_cumulative_occurrence_analysis_diagrams(self):
        processes = ["Designprozess1", "Designprozess2"]
        diagrams = perform_cumulative_occurrence_analysis(processes)
        self.assertEqual(len(diagrams), len(processes))
        for file in diagrams:
            self.assertTrue(os.path.exists(file))
            self.generated_files.append(file)

    def test_markov_chain_analysis_transition_matrix(self):
        data = "dummy_data"
        matrix = perform_markov_chain_analysis(data)
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        self.assertTrue(((matrix.values >= 0) & (matrix.values <= 1)).all())


if __name__ == "__main__":
    unittest.main()
