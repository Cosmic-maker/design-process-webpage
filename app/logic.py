import textwrap

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
from adjustText import adjust_text

matplotlib.use('Agg')
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator

ALLOWED_CODES = {"R", "F", "Be", "Bs", "S", "D"}
DIAGRAM_FOLDER = os.path.join("app", "static", "diagrams")
os.makedirs(DIAGRAM_FOLDER, exist_ok=True)

def process_excel_file(filepath):
    xls = pd.ExcelFile(filepath)
    filename_base = os.path.splitext(os.path.basename(filepath))[0]

    valid_sheets = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        # Prüfe Spaltennamen
        if list(df.columns[:3]) != ["Segment", "Code", "Phase"]:
            raise ValueError(f"Tab '{sheet_name}': Erste drei Spalten müssen 'Segment', 'Code', 'Phase' heißen.")

        # Prüfe 'Segment' auf ganze Zahlen
        if not pd.api.types.is_integer_dtype(df["Segment"]):
            raise ValueError(f"Tab '{sheet_name}': 'Segment' muss aus ganzen Zahlen bestehen.")

        # Prüfe 'Code' auf erlaubte Werte
        if not all(code in ALLOWED_CODES for code in df["Code"].astype(str)):
            raise ValueError(f"Tab '{sheet_name}': 'Code' enthält ungültige Werte.")

        # Prüfe 'Segment'-Reihenfolge
        expected_segments = list(range(1, len(df) + 1))
        if not df["Segment"].tolist() == expected_segments:
            raise ValueError(f"Tab '{sheet_name}': 'Segment' muss von 1 bis n durchnummeriert sein.")

        # Prüfe, dass 'Phase' nicht leer ist
        if df["Phase"].isnull().any() or df["Phase"].astype(str).str.strip().eq("").any():
            raise ValueError(f"Tab '{sheet_name}': 'Phase' darf keine leeren Zellen enthalten.")

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
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from adjustText import adjust_text
    import os

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

    # Analyse-Label
    if analysis_type == 'whole_process':
        combined_df["Analysis_Label"] = combined_df["Register"]
    elif analysis_type == 'process_phases':
        combined_df = combined_df.groupby(["Register", "Phase", "Code"]).size().reset_index(name='Count')
        combined_df["Analysis_Label"] = combined_df.apply(lambda r: f"{r['Register']}\n-\n{r['Phase']}", axis=1)
    else:
        if len(selected_registers) == 1:
            combined_df["Analysis_Label"] = "Segment " + combined_df["Segment"].astype(str)
        else:
            combined_df["Analysis_Label"] = combined_df["Register"] + ": Segment " + combined_df["Segment"].astype(str)

    # Kontingenztabelle und SVD
    contingency_table = pd.crosstab(combined_df["Analysis_Label"], combined_df["Code"])
    X = StandardScaler().fit_transform(contingency_table)
    svd = TruncatedSVD(n_components=2)
    coords = svd.fit_transform(X)

    row_coords = pd.DataFrame(coords, index=contingency_table.index, columns=['Dim1', 'Dim2'])
    col_coords = pd.DataFrame(svd.components_.T, index=contingency_table.columns, columns=['Dim1', 'Dim2'])

    explained_var = svd.explained_variance_ratio_ * 100

    # Plot
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    ax.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax.axvline(0, color='gray', linestyle='--', lw=0.8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Punkte + Texte
    texts = []
    all_points = []

    # Offset, damit Labels initial weiter weg vom Punkt sind
    offset_x = 0.1
    offset_y = 0.1

    # Labels direkt auf Punkt (x, y)
    for code, row in col_coords.iterrows():
        x, y = row['Dim1'], row['Dim2']
        ax.scatter(x, y, color='red', s=15, zorder=2)
        txt = ax.text(x, y, code, fontsize=12, color='red', linespacing=0.5,
                      ha='center', va='center', zorder=4)
        texts.append(txt)
        all_points.append((x, y))

    for label, row in row_coords.iterrows():
        x, y = row['Dim1'], row['Dim2']
        ax.scatter(x, y, color='blue', s=20, zorder=3)
        txt = ax.text(x, y, label, fontsize=11, color='blue', linespacing=0.5,
                      ha='center', va='center', zorder=5)
        texts.append(txt)
        all_points.append((x, y))

    adjust_text(
        texts,
        expand_points=(150, 150),     # sehr großer Abstand zwischen Punkten und Texten
        expand_text=(150, 150),       # sehr großer Abstand zwischen Texten
        force_points=20,              # hohe Kraft, um Abstand zu wahren
        force_text=20,                # hohe Kraft bei Text-Text Abstand
        avoid_points=all_points,      # Punkte meiden
        avoid_text=True,              # Texte meiden
        only_move={'points': 'none', 'text': 'xy'},
        arrowprops=dict(
            arrowstyle='->',
            color='gray',
            lw=0.6,
            shrinkA=0,
            shrinkB=7,
        ),
        precision=0.0001,
        lim=6000,                    # erlaube große Verschiebungen
        autoalign='xy'
    )


    # Achsentitel & Speichern
    ax.set_title(f"Correspondence Analysis: {'Whole Processes' if analysis_type == 'whole_process' else 'Process Phases'}", fontsize=16)
    ax.set_xlabel(f"Dim1 ({explained_var[0]:.1f}%)", fontsize=14)
    ax.set_ylabel(f"Dim2 ({explained_var[1]:.1f}%)", fontsize=14)

    output_dir = os.path.join(os.path.dirname(__file__), 'static', 'diagrams')
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "_".join([reg.replace(" ", "_") for reg in selected_registers])
    output_filename = f"correspondence_{safe_name}_{analysis_type}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, bbox_inches='tight')
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

    ordered_codes = ["R", "F", "Be", "S", "Bs", "D"]
    fbs_results[designprozess_name] = {}
    characterizations[designprozess_name] = {}

    max_segment = df["Segment"].max()

    # Analyse pro Code
    for code in ordered_codes:
        group = cumulative[cumulative["Code"] == code]
        if not group.empty:
            first_occurrence_segment = group["Segment"].min()
            fbs_results[designprozess_name][code] = "Yes" if first_occurrence_segment <= fbs_threshold else "No"
        else:
            fbs_results[designprozess_name][code] = "No"

        x = group["Segment"].values
        y = group["Count"].values

        if len(x) == 0:
            characterizations[designprozess_name][code] = "n.A."
            slopes[code] = None
            continue

        # Linie beginnt bei y=0
        x_extended = np.insert(x, 0, x[0])
        y_extended = np.insert(y, 0, 0)

        # Linie waagrecht bis max_segment weiterführen
        if x_extended[-1] < max_segment:
            x_extended = np.append(x_extended, max_segment)
            y_extended = np.append(y_extended, y_extended[-1])

        # Charakterisierung (mit erweiterten Punkten)
        if len(x_extended) >= min_occurrences_char:
            try:
                a, b, c = np.polyfit(x_extended, y_extended, deg=2)
                threshold = curvature_threshold(max_segment)
                if abs(a) < threshold:
                    curvature_type = "linear"
                elif a > 0:
                    curvature_type = "convex"
                else:
                    curvature_type = "concave"
            except Exception as e:
                print(f"Fehler bei der Charakterisierung für Code {code}: {e}")
                curvature_type = "n.A."
        else:
            curvature_type = "n.A."

        characterizations[designprozess_name][code] = curvature_type

        # Steigung (mit erweiterten Punkten)
        if len(x_extended) >= min_occurrences_slope:
            try:
                slope, _ = np.polyfit(x_extended, y_extended, deg=1)
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

            x_extended = np.insert(x, 0, x[0])
            y_extended = np.insert(y, 0, 0)
            if x_extended[-1] < max_segment:
                x_extended = np.append(x_extended, max_segment)
                y_extended = np.append(y_extended, y_extended[-1])

            plt.plot(x_extended, y_extended, label=code, linewidth=2)

    plt.title(f"Cumulative Occurrence Analysis: {designprozess_name}")
    plt.xlabel("Segment")
    plt.ylabel("Cumulative Count")
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=max_segment)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(title="Code", loc="best")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(diagram_folder, f"{filename_base}_{sheet_name}_cumulative.png")
    # Plot-Limits erweitern, damit Labels am Rand nicht abgeschnitten werden
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    x_range = x_max - x_min
    y_range = y_max - y_min

    plt.xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    plt.savefig(output_path)
    plt.close()

    # Slope Barchart
    plt.figure(figsize=(10, 6))
    slope_values = [slopes.get(code) for code in ordered_codes]
    bars = [val if val is not None else 0 for val in slope_values]
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

def curvature_threshold(max_segment):
    return max(0.0003, 0.008 - (max_segment / 1000))


def perform_markov_chain_analysis(df, sheet_name, filename_base, threshold=0.0, action="generate"):
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

    # Nur die dominanten Übergänge behalten, wenn show_dominant ausgewählt wurde
    if action == "show_dominant":
        # Neue Logik: Für jeden Zielzustand nur den Übergang mit höchster Wahrscheinlichkeit behalten
        dominant_transitions = {}

        # Erst alle möglichen Übergänge sammeln
        for from_code in transition_probs:
            for to_code, prob in transition_probs[from_code].items():
                if to_code not in dominant_transitions:
                    dominant_transitions[to_code] = (from_code, prob)
                else:
                    # Nur behalten wenn Wahrscheinlichkeit höher ist
                    if prob > dominant_transitions[to_code][1]:
                        dominant_transitions[to_code] = (from_code, prob)

        # Übergangsmatrix neu aufbauen
        new_transition_probs = {from_code: {} for from_code in ALLOWED_CODES}
        for to_code, (from_code, prob) in dominant_transitions.items():
            new_transition_probs[from_code][to_code] = prob

        transition_probs = new_transition_probs

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
                'D': (3, 1)
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

        title = f"Markov Chain: {sheet_name}"
        if action == "show_dominant":
            title += " (dominant transitions only)"
        else:
            title += f" (threshold: {threshold*100:.0f}%)"
        plt.title(title)

        markov_dir = os.path.join(DIAGRAM_FOLDER, "markov")
        os.makedirs(markov_dir, exist_ok=True)
        output_path = os.path.join(markov_dir, f"{sheet_name}_markov_{threshold}_{action if action in ['show_dominant'] else 'normal'}.png")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    except Exception as e:
        print(f"Fehler bei der Diagrammerstellung: {e}")
    finally:
        plt.close('all')