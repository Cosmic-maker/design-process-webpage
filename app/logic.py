import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')
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

        valid_sheets.append((filename_base, df))

    return valid_sheets

def create_combined_excel(sheet_dict, output_path):

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_exists = os.path.exists(output_path)

    # Open the Excel file (write mode if new, append mode if existing)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a' if file_exists else 'w') as writer:
        for sheet_name, df in sheet_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def perform_correspondence_analysis(combined_file, selected_registers, analysis_type='process_phases'):
    from collections import defaultdict

    if not selected_registers:
        raise ValueError("No registers selected")

    if analysis_type == 'whole_process' and len(selected_registers) < 2:
        raise ValueError("Whole process analysis requires at least 2 processes")

    # Daten laden
    data_frames = []
    for reg in selected_registers:
        df = pd.read_excel(combined_file, sheet_name=reg)
        df["Register"] = reg
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)

    # Labels erstellen
    if analysis_type == 'whole_process':
        combined_df["Analysis_Label"] = combined_df["Register"]
    else:
        if len(selected_registers) == 1:
            combined_df["Analysis_Label"] = "Segment " + combined_df["Segment"].astype(str)
        else:
            combined_df["Analysis_Label"] = combined_df["Register"] + ": Segment " + combined_df["Segment"].astype(str)

    # Kontingenztabelle
    contingency_table = pd.crosstab(combined_df["Analysis_Label"], combined_df["Code"])

    # Standardisieren + SVD
    X = StandardScaler().fit_transform(contingency_table)
    svd = TruncatedSVD(n_components=2)
    coords = svd.fit_transform(X)

    row_coords = pd.DataFrame(coords, index=contingency_table.index, columns=['Dim1', 'Dim2'])
    col_coords = pd.DataFrame(svd.components_.T, index=contingency_table.columns, columns=['Dim1', 'Dim2'])

    # Erklärte Varianz in Prozent
    explained_var = svd.explained_variance_ratio_ * 100

    # Farben (mehr Farben für viele Register)
    unique_registers = list(dict.fromkeys(combined_df["Register"]))
    base_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'pink', 'cyan', 'magenta', 'yellow']
    color_map = {reg: base_colors[i % len(base_colors)] for i, reg in enumerate(unique_registers)}

    plt.figure(figsize=(12, 8))

    # === Analyse-Labels gruppieren nach gerundeten Koordinaten ===
    coord_groups = defaultdict(list)
    for label in row_coords.index:
        x = round(row_coords.loc[label, 'Dim1'], 5)
        y = round(row_coords.loc[label, 'Dim2'], 5)
        coord_groups[(x, y)].append(label)

    # Punkte und Beschriftungen der Analyse-Labels
    for (x, y), labels in coord_groups.items():
        # Bestimme Farbe nach Register des ersten Labels
        first_label = labels[0]
        if ":" in first_label:
            reg = first_label.split(":")[0].strip()
        else:
            reg = combined_df[combined_df["Analysis_Label"] == first_label]["Register"].iloc[0]

        plt.scatter(x, y, color=color_map.get(reg, 'black'), s=50, marker='o')

        # Wenn nur ein Label, normal beschriften
        if len(labels) == 1:
            # Beschriftung ohne „Segment“
            import re
            lab = labels[0]
            new_label = re.sub(r'Segment\s*', '', lab)
            plt.annotate(new_label,
                         (x, y),
                         fontsize=8,
                         color=color_map.get(reg, 'black'),
                         xytext=(5, 5),
                         textcoords='offset points',
                         ha='left', va='bottom')
        else:
            # Mehrere Labels: Segmentnummern extrahieren ohne "Segment" und prefix ohne Segmentnummer
            segments = []
            prefix = None
            import re
            for lab in labels:
                m = re.search(r'Segment\s*(\d+)', lab)
                if m:
                    segments.append(m.group(1))
                    prefix_candidate = lab[:m.start()].strip()
                    if prefix is None:
                        prefix = prefix_candidate
                else:
                    segments.append(lab)
                    prefix = None

            if prefix:
                label_text = f"{prefix}: {','.join(segments)}"
            else:
                label_text = ",".join(segments)

            plt.annotate(label_text,
                         (x, y),
                         fontsize=8,
                         color=color_map.get(reg, 'black'),
                         xytext=(5, 5),
                         textcoords='offset points',
                         ha='left', va='bottom')

    # === Codes gruppieren ===
    code_groups = defaultdict(list)
    for code in col_coords.index:
        x = round(col_coords.loc[code, 'Dim1'], 5)
        y = round(col_coords.loc[code, 'Dim2'], 5)
        code_groups[(x, y)].append(code)

    # Punkte für Codes (rot) und Beschriftungen
    for (x, y), codes in code_groups.items():
        plt.scatter(x, y, color='red', marker='o', s=30)
        label_text = ",".join(codes)
        plt.annotate(label_text,
                     (x, y),
                     fontsize=10,
                     color='red',
                     xytext=(5, 5),
                     textcoords='offset points',
                     ha='left', va='bottom')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')

    plt.title(f"Correspondence Analysis: {'Whole Processes' if analysis_type == 'whole_process' else 'Process Phases'}")
    plt.xlabel(f"Dim1 ({explained_var[0]:.1f}%)")
    plt.ylabel(f"Dim2 ({explained_var[1]:.1f}%)")

    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=reg,
                                   markerfacecolor=col, markersize=8)
                        for reg, col in color_map.items()])
    plt.grid(True)

    # Speichern
    output_dir = os.path.join(os.path.dirname(__file__), 'static', 'diagrams')
    os.makedirs(output_dir, exist_ok=True)

    safe_name = "_".join([reg.replace(" ", "_") for reg in selected_registers])
    output_filename = f"correspondence_{safe_name}_{analysis_type}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path)
    plt.close()

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

    # Nur die gewünschten Codes in der vorgegebenen Reihenfolge
    ordered_codes = ["R", "F", "Be", "S", "Bs", "D"]
    all_codes = [code for code in ordered_codes if code in df["Code"].unique()]

    fbs_results[designprozess_name] = {}
    characterizations[designprozess_name] = {}

    for code in ordered_codes:
        group = cumulative[cumulative["Code"] == code]

        if not group.empty:
            first_occurrence_segment = group["Segment"].min()
            fbs_results[designprozess_name][code] = "Yes" if first_occurrence_segment <= fbs_threshold else "No"
        else:
            fbs_results[designprozess_name][code] = "No"

        x = group["Segment"]
        y = group["Count"]

        # Charakterisierung
        if len(group) >= min_occurrences_char:
            try:
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

        # Steigung
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
    for code in ordered_codes:
        group = cumulative[cumulative["Code"] == code]
        if not group.empty:
            x = group["Segment"].values
            y = group["Count"].values

            # Linie beginnt bei y=0 (selbes Segment wie erstes echtes Vorkommen)
            x_extended = np.insert(x, 0, x[0])  # z.B. [3, 3, 4, 5]
            y_extended = np.insert(y, 0, 0)     # z.B. [0, 1, 2, 3]

            plt.plot(x_extended, y_extended, label=code, linewidth=2)

    plt.title(f"Cumulative Occurrence Analysis: {designprozess_name}")
    plt.xlabel("Segment")
    plt.ylabel("Cumulative Count")
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    # Nur Integer-Ticks auf beiden Achsen
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(title="Code", loc="best")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(diagram_folder, f"{filename_base}_{sheet_name}_cumulative.png")
    plt.savefig(output_path)
    plt.close()

    # Slope Barchart
    plt.figure(figsize=(10, 6))
    slope_values = [slopes.get(code) for code in ordered_codes]
    bars = []

    for code, val in zip(ordered_codes, slope_values):
        if val is not None:
            bars.append(val)
        else:
            bars.append(0)

    bar_colors = ['skyblue' if val is not None else 'white' for val in slope_values]
    edge_colors = ['black' for _ in slope_values]

    plt.bar(ordered_codes, bars, color=bar_colors, edgecolor=edge_colors)
    plt.title(f"Slope Analysis: {designprozess_name}")
    plt.xlabel("Code")
    plt.ylabel("Slope")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(diagram_folder, f"{filename_base}_{sheet_name}_slopes.png"))
    plt.close()

    return {
        "fbs_results": fbs_results,
        "characterizations": characterizations,
    }


def perform_markov_chain_analysis(df, sheet_name, filename_base, threshold=0.0):
    codes = df["Code"].astype(str).tolist()
    transitions = list(zip(codes[:-1], codes[1:]))
    transition_counts = {from_code: {to_code: 0 for to_code in ALLOWED_CODES} for from_code in ALLOWED_CODES}
    for from_code, to_code in transitions:
        if from_code in ALLOWED_CODES and to_code in ALLOWED_CODES:
            transition_counts[from_code][to_code] += 1

    transition_probs = {}
    for from_code in ALLOWED_CODES:
        total_transitions = sum(transition_counts[from_code].values())
        if total_transitions == 0:
            continue

        transition_probs[from_code] = {}
        for to_code in ALLOWED_CODES:
            prob = transition_counts[from_code][to_code] / total_transitions
            if prob >= threshold:
                transition_probs[from_code][to_code] = prob

    G = nx.DiGraph()

    for code in ALLOWED_CODES:
        if code in transition_probs or any(code in v for v in transition_probs.values()):
            G.add_node(code, size=1000)

    for from_code in transition_probs:
        for to_code, prob in transition_probs[from_code].items():
            if prob > 0:
                G.add_edge(from_code, to_code, weight=prob)

    # Visualisierung
    try:
        plt.figure(figsize=(8, 8))

        if len(G.nodes()) == 0:
            plt.text(0.5, 0.5, "Keine Übergänge über der Schwelle",
                     ha='center', va='center')
        else:
            fixed_positions = {
                'R': (0, 1),
                'F': (1, 1),
                'Be': (1, 0),
                'S': (2, 1),
                'Bs': (2, 0),
                'D': (2, -1)
            }
            pos = {k: fixed_positions[k] for k in G.nodes() if k in fixed_positions}

            nx.draw_networkx_nodes(G, pos, node_size=3000,
                                   node_color='skyblue', edgecolors="black", alpha=0.7)

            edges = G.edges()
            weights = [G[u][v]["weight"] for u, v in edges]
            nx.draw_networkx_edges(
                G, pos, edgelist=edges,
                width=[weight * 10 for weight in weights],
                alpha=0.7, edge_color='grey', arrows=True,
                arrowstyle='-|>', min_source_margin=25,
                min_target_margin=25, connectionstyle='arc3,rad=0.1'
            )

            edge_labels = {(u, v): f"{w:.2f}" for u, v, w in G.edges(data='weight')}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            nx.draw_networkx_labels(G, pos, font_size=14,
                                    font_weight='bold', font_color='black')

        plt.title(f"Markov Chain: {sheet_name} (Schwelle: {threshold*100:.0f}%)")

        markov_dir = os.path.join(DIAGRAM_FOLDER, "markov")
        os.makedirs(markov_dir, exist_ok=True)
        output_path = os.path.join(markov_dir, f"{sheet_name}_markov_{threshold}.png")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    except Exception as e:
        print(f"Fehler bei der Diagrammerstellung: {e}")
    finally:
        plt.close('all')