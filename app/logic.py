import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import prince
import os


ALLOWED_CODES = {"R", "F", "Be", "Bs", "S", "D"}
DIAGRAM_FOLDER = "diagramme"
os.makedirs(DIAGRAM_FOLDER, exist_ok=True)

def process_excel_file(filepath):
    xls = pd.ExcelFile(filepath)
    filename_base = os.path.splitext(os.path.basename(filepath))[0]

    valid_sheets = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        if list(df.columns[:2]) != ["Segment", "Code"]:
            raise ValueError(f"Tab '{sheet_name}': Spalten müssen 'Segment' und 'Code' heißen.")

        if not pd.api.types.is_integer_dtype(df["Segment"]):
            raise ValueError(f"Tab '{sheet_name}': 'Segment' muss aus ganzen Zahlen bestehen.")

        if not all(code in ALLOWED_CODES for code in df["Code"].astype(str)):
            raise ValueError(f"Tab '{sheet_name}': 'Code' enthält ungültige Werte.")

        expected_segments = list(range(1, len(df) + 1))
        if not df["Segment"].tolist() == expected_segments:
            raise ValueError(f"Tab '{sheet_name}': 'Segment' muss von 1 bis n durchnummeriert sein.")

        # ✅ Append als Liste von Tupeln
        valid_sheets.append((filename_base, df))

    return valid_sheets

def create_combined_excel(sheet_dict, output_path):
    # Check if the directory exists, if not create it
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the file exists or not
    file_exists = os.path.exists(output_path)

    # Open the Excel file (write mode if new, append mode if existing)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a' if file_exists else 'w') as writer:
        for sheet_name, df in sheet_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


#Korrespondenzanalyse wird noch nicht verwendet
def perform_correspondence_analysis(df, sheet_name, filename_base):
    # ✅ 4. Kreuztabelle & CA
    contingency_table = pd.crosstab(df['Segment'].astype(str), df['Code'])

    ca = prince.CA(n_components=2)
    ca = ca.fit(contingency_table)

    row_coords = ca.row_coordinates(contingency_table)
    col_coords = ca.column_coordinates(contingency_table)

    # ✅ 5. Diagramm
    plt.figure(figsize=(8, 6))
    plt.scatter(row_coords[0], row_coords[1], color='blue', label='Segmente')
    plt.scatter(col_coords[0], col_coords[1], color='red', label='Codes')

    for i, label in enumerate(row_coords.index):
        plt.annotate(label, (row_coords.iloc[i, 0], row_coords.iloc[i, 1]), color='blue')

    for i, label in enumerate(col_coords.index):
        plt.annotate(label, (col_coords.iloc[i, 0], col_coords.iloc[i, 1]), color='red')

    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.title(f"Korrespondenzanalyse: {sheet_name}")
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(DIAGRAM_FOLDER, f"{filename_base}_{sheet_name}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"📊 Diagramm gespeichert: {output_path}")


def perform_markov_chain_analysis(df, sheet_name, filename_base):
    # Codes extrahieren (hier als Beispiel)
    codes = df["Code"].astype(str).tolist()
    transitions = list(zip(codes[:-1], codes[1:]))

    # Übergangshäufigkeit berechnen
    transition_counts = {from_code: {to_code: 0 for to_code in ALLOWED_CODES} for from_code in ALLOWED_CODES}
    for from_code, to_code in transitions:
        if from_code in ALLOWED_CODES and to_code in ALLOWED_CODES:
            transition_counts[from_code][to_code] += 1

    # Wahrscheinlichkeiten berechnen
    transition_probs = {from_code: {to_code: 0 for to_code in ALLOWED_CODES} for from_code in ALLOWED_CODES}
    for from_code in ALLOWED_CODES:
        total_transitions = sum(transition_counts[from_code].values())
        for to_code in ALLOWED_CODES:
            if total_transitions > 0:
                transition_probs[from_code][to_code] = transition_counts[from_code][to_code] / total_transitions

    G = nx.DiGraph()

    for code in ALLOWED_CODES:
        G.add_node(code, size=1000)


    for from_code in ALLOWED_CODES:
        for to_code in ALLOWED_CODES:
            prob = transition_probs[from_code][to_code]
            if prob > 0:
                G.add_edge(from_code, to_code, weight=prob)

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', edgecolors="black", alpha=0.7)

    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=[weight * 10 for weight in weights],
        alpha=0.7,
        edge_color='grey',
        arrows=True,
        arrowstyle='-|>',
        min_source_margin=25,
        min_target_margin=25,
        connectionstyle='arc3,rad=0.1'
    )
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black')

    plt.title(f"Markov Chain Analysis: {sheet_name}")

    diagram_folder = os.path.join(os.getcwd(), 'app', 'diagrams')
    os.makedirs(diagram_folder, exist_ok=True)
    output_path = os.path.join(diagram_folder, f"{sheet_name}_markov.png")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"🔁 Markov-Diagramm gespeichert: {output_path}")
