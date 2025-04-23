import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prince
import os

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

ALLOWED_CODES = {"R", "F", "Be", "Bs", "S", "D"}
DIAGRAM_FOLDER = os.path.join("app", "static", "diagrams")
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
def perform_correspondence_analysis(combined_file, selected_registers):
    data_frames = {}

    # 1. Lade ausgewählte Sheets und füge "Register"-Spalte hinzu
    for reg in selected_registers:
        df = pd.read_excel(combined_file, sheet_name=reg)
        df["Register"] = reg
        data_frames[reg] = df

    # 2. Kombiniere alle DataFrames
    combined_df = pd.concat(data_frames.values(), ignore_index=True)

    # 3. Erstelle eindeutige Segmentbezeichnungen z. B. "Brot: Segment1"
    combined_df["Segment_Label"] = combined_df["Register"] + ": " + combined_df["Segment"].astype(str)

    # 4. Erstelle Kreuztabelle (Kontingenztabelle)
    contingency_table = pd.crosstab(combined_df["Segment_Label"], combined_df["Code"])

    # 5. Standardisiere Daten für CA
    X = StandardScaler().fit_transform(contingency_table)

    # 6. Verwende SVD zur Dimensionsreduktion
    svd = TruncatedSVD(n_components=2)
    coords = svd.fit_transform(X)

    # 7. Koordinaten für Segment-Zeilen und Code-Spalten
    row_coords = pd.DataFrame(coords, index=contingency_table.index)
    col_coords = pd.DataFrame(svd.components_.T, index=contingency_table.columns)

    # 8. Mapping von Segment zu Register für Farbcodierung
    segment_to_register = {label: label.split(":")[0] for label in contingency_table.index}
    color_map = {
        reg: color for reg, color in zip(selected_registers, ['blue', 'green', 'orange', 'purple', 'brown'])
    }

    # 9. Diagramm zeichnen
    plt.figure(figsize=(10, 8))

    for i, label in enumerate(row_coords.index):
        reg = segment_to_register[label]
        color = color_map.get(reg, 'gray')
        plt.scatter(row_coords.iloc[i, 0], row_coords.iloc[i, 1], color=color, label=reg if i == 0 or reg not in row_coords.index[:i].map(segment_to_register) else "")
        plt.annotate(label, (row_coords.iloc[i, 0], row_coords.iloc[i, 1]), fontsize=8, color=color)

    # Codes (Spalten)
    plt.scatter(col_coords[0], col_coords[1], color='red', label='Codes')
    for i, label in enumerate(col_coords.index):
        plt.annotate(label, (col_coords.iloc[i, 0], col_coords.iloc[i, 1]), fontsize=8, color='red')

    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.title(f"Korrespondenzanalyse: {', '.join(selected_registers)}")
    plt.legend()
    plt.grid(True)

    # 10. Sicherer Dateiname + Pfad
    safe_filename = "_".join([reg.replace(" ", "_") for reg in selected_registers])
    output_filename = f"{safe_filename}_combined_correspondence.png"
    output_path = os.path.join("app", "static", "diagrams", output_filename)

    plt.savefig(output_path)
    plt.close()

    print(f"📊 Diagramm gespeichert: {output_path}")
    return output_filename


def perform_cumulative_occurence_analysis(df, sheet_name, filename_base, min_occurrences_char, min_occurrences_slope, fbs_threshold):
    diagram_folder = os.path.join("app", "static", "diagrams", "cumulative_occurrence_analysis")
    os.makedirs(diagram_folder, exist_ok=True)

    df["Count"] = 1
    cumulative = df.groupby(["Code", "Segment"]).count().groupby(level=0).cumsum().reset_index()

    slopes = {}
    characterizations = {}
    fbs_results = {}

    designprozess_name = filename_base
    print(f"Verarbeite Designprozess: '{designprozess_name}'")

    all_codes = df["Code"].unique()
    fbs_results[designprozess_name] = {}
    characterizations[designprozess_name] = {}

    for code in all_codes:
        group = cumulative[cumulative["Code"] == code]

        if not group.empty:
            # First occurrence at start
            first_occurrence_segment = group["Segment"].min()
            fbs_results[designprozess_name][code] = "Yes" if first_occurrence_segment <= fbs_threshold else "No"
        else:
            fbs_results[designprozess_name][code] = "No"

        x = group["Segment"]
        y = group["Count"]

        # Charakterisierung (min. Anzahl an Occurrences)
        if len(group) >= min_occurrences_char:
            try:
                # Charakterisierung über quadratische Regression
                a, b, c = np.polyfit(x, y, deg=2)
                if abs(a) < 0.01:
                    curvature_type = "linear"
                elif a > 0:
                    curvature_type = "convex"
                else:
                    curvature_type = "concave"
            except Exception as e:
                print(f"Fehler bei der Charakterisierung für Code {code}: {e}")
                curvature_type = "unbekannt"
        else:
            curvature_type = "unbekannt"

        characterizations[designprozess_name][code] = curvature_type

        # Steigung (nur wenn min. Voraussetzung erfüllt)
        if len(group) >= min_occurrences_slope:
            try:
                slope, _ = np.polyfit(x, y, deg=1)
                slopes[code] = slope
            except Exception as e:
                print(f"Fehler bei der Steigungsberechnung für Code {code}: {e}")
                slopes[code] = None
        else:
            slopes[code] = None

    # Cumulative Occurrence Plot
    plt.figure(figsize=(10, 6))
    for code in all_codes:
        group = cumulative[cumulative["Code"] == code]
        if not group.empty:
            plt.plot(group["Segment"], group["Count"], label=f"{code}")
    plt.title(f"Cumulative Occurrence Analysis: {designprozess_name}")
    plt.xlabel("Segment")
    plt.ylabel("Kumulative Häufigkeit")
    plt.legend(title="Code", loc="best")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(diagram_folder, f"{filename_base}_{sheet_name}_cumulative.png")
    plt.savefig(output_path)
    plt.close()

    # Slope Barchart
    plt.figure(figsize=(10, 6))
    filtered_codes = [code for code in all_codes if slopes.get(code) is not None]
    slope_values = [slopes[code] for code in filtered_codes]

    plt.bar(filtered_codes, slope_values, color='skyblue', edgecolor='black')
    plt.title(f"Slope Analysis: {designprozess_name}")
    plt.xlabel("Code")
    plt.ylabel("Steigung")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(diagram_folder, f"{filename_base}_{sheet_name}_slopes.png"))
    plt.close()

    return {
        "fbs_results": fbs_results,
        "characterizations": characterizations,
    }

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

    DIAGRAM_FOLDER = os.path.join(os.getcwd(), 'app', 'static', "diagrams")
    os.makedirs(DIAGRAM_FOLDER, exist_ok=True)
    output_path = os.path.join(DIAGRAM_FOLDER, f"{sheet_name}_markov.png")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"🔁 Markov-Diagramm gespeichert: {output_path}")