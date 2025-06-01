import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt  # Nur notwendig, wenn du wirklich speicherst

# Angenommen, deine Logikfunktionen heißen z. B.:
# from your_module import analyse_prozesse, erstelle_diagramm, berechne_markov_matrix

class TestLogic(unittest.TestCase):

    def test_process_selection(self):
        """Teste, ob 1-N Prozesse ausgewählt werden können"""
        selected_processes = ["Designprozess1", "Designprozess2", "Designprozess3"]

        # Hier sollte deine echte Analysefunktion stehen:
        # processed_processes = analyse_prozesse(selected_processes)
        processed_processes = selected_processes.copy()  # Simulierte Verarbeitung

        self.assertEqual(len(processed_processes), 3)
        self.assertIn("Designprozess2", processed_processes)

    @patch('matplotlib.pyplot.savefig')  # Verhindert tatsächliches Speichern
    def test_diagram_creation(self, mock_save):
        """Teste die Diagrammerstellung pro Prozess"""
        processes = ["Designprozess1", "Designprozess2"]

        created_diagrams = []
        for process in processes:
            # Hier sollte dein echter Funktionsaufruf stehen:
            # pfad = erstelle_diagramm(process)
            pfad = f"diagram_{process}.png"
            created_diagrams.append(pfad)

        self.assertEqual(len(created_diagrams), 2)
        self.assertTrue(all(d.startswith("diagram_") for d in created_diagrams))

    def test_markov_matrix(self):
        """Teste die Markov-Transitionsmatrix"""
        test_transitions = [
            ["R", "F"],
            ["F", "Be"],
            ["Be", "S"]
        ]

        # Hier sollte deine echte Matrix-Berechnung stehen:
        # calculated_matrix = berechne_markov_matrix(test_transitions)
        calculated_matrix = {
            "R": {"F": 1.0},
            "F": {"Be": 1.0},
            "Be": {"S": 1.0}
        }

        self.assertEqual(calculated_matrix["R"]["F"], 1.0)
        self.assertEqual(len(calculated_matrix), 3)

if __name__ == '__main__':
    unittest.main()
