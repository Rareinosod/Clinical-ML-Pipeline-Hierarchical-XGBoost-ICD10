import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CARTELLA_RISULTATI = BASE_DIR / "risultati_finali"
OUT_DIR = BASE_DIR / "Analisi_Comparativa"

def costruisci_master_dataframe():
    """Scansiona tutti gli esperimenti e calcola le medie globali e per classe, gestendo il Fallback."""
    dati_aggregati = []
    
    if not CARTELLA_RISULTATI.exists():
        print(f"Cartella {CARTELLA_RISULTATI} non trovata.")
        return pd.DataFrame()

    for nome_esp in os.listdir(CARTELLA_RISULTATI):
        cartella_esp = os.path.join(CARTELLA_RISULTATI, nome_esp)
        if not os.path.isdir(cartella_esp) or not nome_esp.startswith("Run_"):
            continue
            
        nome_esperimento = nome_esp.split("_Esperimento")[-1]
        
        validazione = "Sconosciuta"
        if "KFold" in nome_esp: validazione = "K-Fold"
        elif "Bootstrap" in nome_esp: validazione = "Bootstrap"
        else: validazione = "Holdout"

        for nome_modello_dir in os.listdir(cartella_esp):
            cartella_modello = os.path.join(cartella_esp, nome_modello_dir)
            if os.path.isdir(cartella_modello) and nome_modello_dir.startswith("Risultati_"):
                
                nome_base = nome_modello_dir.replace("Risultati_", "")
                
                is_fallback = "Fallback" in nome_base
                base_pulito = nome_base.replace("_Fallback", "")
                
                metrica_tuning = "F1" 
                if base_pulito.endswith("_F1"):
                    metrica_tuning = "F1"
                    nome_modello = base_pulito[:-3]
                elif base_pulito.endswith("_F2"):
                    metrica_tuning = "F2"
                    nome_modello = base_pulito[:-3]
                else:
                    nome_modello = base_pulito
                
                label_modello = f"{nome_modello}_{metrica_tuning}"
                if is_fallback: label_modello += "_Fallback"
                
                # --- ESTRAZIONE PER-CLASS SCORE DAI REPORT TXT ---
                f_class_0, f_class_1, f_class_2 = [], [], []
                for root, dirs, files in os.walk(cartella_modello):
                    for file in files:
                        if file.startswith("report_") and file.endswith(".txt"):
                            try:
                                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    c0 = re.search(r'Classe 0:\s*([0-9.]+)', content)
                                    c1 = re.search(r'Classe 1:\s*([0-9.]+)', content)
                                    c2 = re.search(r'Classe 2:\s*([0-9.]+)', content)
                                    if c0: f_class_0.append(float(c0.group(1)))
                                    if c1: f_class_1.append(float(c1.group(1)))
                                    if c2: f_class_2.append(float(c2.group(1)))
                            except Exception:
                                pass
                
                media_c0 = np.mean(f_class_0) if f_class_0 else np.nan
                media_c1 = np.mean(f_class_1) if f_class_1 else np.nan
                media_c2 = np.mean(f_class_2) if f_class_2 else np.nan

                csv_file = os.path.join(cartella_modello, f"risultati_per_compartimento_{nome_modello}.csv")
                if os.path.exists(csv_file):
                    df_modello = pd.read_csv(csv_file)
                    
                    col_f1 = "F1_macro" if "F1_macro" in df_modello.columns else ("Macro-F1" if "Macro-F1" in df_modello.columns else None)
                    col_f2 = "F_Target_macro" if "F_Target_macro" in df_modello.columns and metrica_tuning == "F2" else ("F2_macro" if "F2_macro" in df_modello.columns else None)
                    col_acc = "Accuracy"
                    
                    media_f1 = df_modello[col_f1].mean() if col_f1 else np.nan
                    std_f1 = df_modello[col_f1].std() if (col_f1 and len(df_modello) > 1) else 0.0
                    media_f2 = df_modello[col_f2].mean() if col_f2 else np.nan
                    std_f2 = df_modello[col_f2].std() if (col_f2 and len(df_modello) > 1) else 0.0
                    media_acc = df_modello[col_acc].mean() if col_acc in df_modello.columns else np.nan
                    
                    dati_aggregati.append({
                        "Esperimento": f"Exp {nome_esperimento.split('_')[0]}",
                        "Setup": "_".join(nome_esperimento.split('_')[1:]),
                        "Validazione": validazione,
                        "Modello_Base": nome_modello,
                        "Metrica_Tuning": metrica_tuning,
                        "Is_Fallback": is_fallback,
                        "Modello_Label": label_modello,
                        "F1_Mean": media_f1,
                        "F1_Std": std_f1,
                        "F2_Mean": media_f2,
                        "F2_Std": std_f2,
                        "Accuracy": media_acc,
                        "F_Class_0": media_c0,
                        "F_Class_1": media_c1,
                        "F_Class_2": media_c2
                    })
                    
    return pd.DataFrame(dati_aggregati)

def raccogli_feature_importance():
    dati_fi = []
    if not CARTELLA_RISULTATI.exists(): return pd.DataFrame()

    for nome_esp in os.listdir(CARTELLA_RISULTATI):
        cartella_esp = os.path.join(CARTELLA_RISULTATI, nome_esp)
        if not os.path.isdir(cartella_esp) or not nome_esp.startswith("Run_"): continue

        for nome_modello_dir in os.listdir(cartella_esp):
            cartella_modello = os.path.join(cartella_esp, nome_modello_dir)
            if os.path.isdir(cartella_modello) and nome_modello_dir.startswith("Risultati_"):
                
                file_fi = None
                for file_name in os.listdir(cartella_modello):
                    if file_name.endswith(".csv") and ("importance" in file_name.lower() or "importanza" in file_name.lower()):
                        file_fi = os.path.join(cartella_modello, file_name)
                        break
                
                if file_fi:
                    try:
                        df_fi = pd.read_csv(file_fi)
                        col_feature = next((col for col in df_fi.columns if col.lower() in ['feature', 'variabile', 'name', 'nome']), None)
                        if col_feature:
                            col_imp = next((col for col in df_fi.columns if any(k in col.lower() for k in ['importance', 'importanza', 'mean', 'weight'])), None)
                            if col_imp: df_fi = df_fi.sort_values(by=col_imp, ascending=False)
                            
                            top_features = df_fi[col_feature].head(15).tolist()
                            for rank, feat in enumerate(top_features, start=1):
                                dati_fi.append({
                                    "Esperimento": nome_esp,
                                    "Modello": nome_modello_dir.replace("Risultati_", ""),
                                    "Feature": feat,
                                    "Rank": rank
                                })
                    except Exception: pass
    return pd.DataFrame(dati_fi)

def genera_report_e_grafici(df_perf, df_fi):
    if df_perf.empty:
        print("Nessun dato di performance trovato.")
        return
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(OUT_DIR)    
    
    ha_f1 = df_perf["F1_Mean"].notna().any()
    ha_f2 = df_perf["F2_Mean"].notna().any()

    # Esportazione CSV Completo
    df_perf.sort_values(by=["Esperimento", "F1_Mean"], ascending=[True, False]).to_csv(os.path.join(out_dir_str, "master_report_completo.csv"), index=False)
    
    # Feature Importance
    df_fi_globale = pd.DataFrame()
    if not df_fi.empty:
        conteggio_fi = df_fi['Feature'].value_counts().reset_index()
        conteggio_fi.columns = ['Feature', 'Frequenza_nella_Top_15']
        rank_medio = df_fi.groupby('Feature')['Rank'].mean().reset_index()
        rank_medio.columns = ['Feature', 'Rank_Medio']
        df_fi_globale = pd.merge(conteggio_fi, rank_medio, on='Feature').sort_values(by=['Frequenza_nella_Top_15', 'Rank_Medio'], ascending=[False, True])
        df_fi_globale.to_csv(os.path.join(out_dir_str, "classifica_Feature_Importance_Globale.csv"), index=False)

    print("\n" + "="*70)
    print(" 🏆 RISULTATI ANALISI COMPARATIVA 🏆")
    print("="*70)
    
    with open(os.path.join(out_dir_str, "REPORT_VINCITORI_E_CLASSIFICHE.txt"), "w", encoding="utf-8") as f:
        
        # Top F1: Ordinamento primario Mean (Decrescente), secondario Std (Crescente)
        if ha_f1:
            df_f1_puro = df_perf[df_perf['Metrica_Tuning'] == 'F1'].sort_values(by=['F1_Mean', 'F1_Std'], ascending=[False, True])
            if not df_f1_puro.empty:
                df_f1_puro.to_csv(os.path.join(out_dir_str, "classifica_F1.csv"), index=False)
                titolo = "🏆 TOP 3 MODELLI F1 (Massima Precisione Globale) 🏆"
                print(f"\n{titolo}")
                f.write(f"{titolo}\n===========================================\n")
                for i, row in df_f1_puro.head(3).iterrows():
                    testo = f"{row['Modello_Label']} | F1: {row['F1_Mean']:.4f} (±{row['F1_Std']:.4f}) | {row['Esperimento']} ({row['Setup']})"
                    print(testo)
                    f.write(f"{testo}\n")
                f.write("\n")

        # Top F2: Ordinamento primario Mean (Decrescente), secondario Std (Crescente)
        if ha_f2:
            df_f2_valid = df_perf.dropna(subset=['F2_Mean'])
            df_f2_puro = df_f2_valid[df_f2_valid['Metrica_Tuning'] == 'F2'].sort_values(by=['F2_Mean', 'F2_Std'], ascending=[False, True])
            
            if not df_f2_puro.empty:
                df_f2_puro.to_csv(os.path.join(out_dir_str, "classifica_F2.csv"), index=False)
                titolo = "🏆 TOP 3 MODELLI F2 (Ottimizzati per Sensibilità Clinica) 🏆"
                print(f"\n{titolo}")
                f.write(f"{titolo}\n===========================================\n")
                for i, row in df_f2_puro.head(3).iterrows():
                    testo = f"{row['Modello_Label']} | F2: {row['F2_Mean']:.4f} (±{row['F2_Std']:.4f}) | {row['Esperimento']} ({row['Setup']})"
                    print(testo)
                    f.write(f"{testo}\n")
                f.write("\n")

        # Bilanciati: Filtro Mean_F1, Ordinamento primario Mean_F2 (Decrescente), secondario Std_F2 (Crescente)
        df_bilanciati = pd.DataFrame()
        if ha_f2:
            soglia_f1 = df_f2_valid['F1_Mean'].median() 
            df_bilanciati = df_f2_valid[df_f2_valid['F1_Mean'] >= soglia_f1].sort_values(by=['F2_Mean', 'F2_Std'], ascending=[False, True])
            
            if not df_bilanciati.empty:
                df_bilanciati.to_csv(os.path.join(out_dir_str, "classifica_Scelta_Bilanciata.csv"), index=False)
                titolo = f"⚖️ TOP 3 SCELTE BILANCIATE (Miglior F2 con F1 >= {soglia_f1:.4f}) ⚖️"
                print(f"\n{titolo}")
                f.write(f"{titolo}\n===========================================\n")
                for i, row in df_bilanciati.head(3).iterrows():
                    testo = f"{row['Modello_Label']} | F2: {row['F2_Mean']:.4f} (±{row['F2_Std']:.4f}) (F1: {row['F1_Mean']:.4f}) | {row['Esperimento']} ({row['Setup']})"
                    print(testo)
                    f.write(f"{testo}\n")
                f.write("\n")

        # Feature Importance Testuale
        if not df_fi.empty:
            titolo = "🧬 TOP 10 FEATURE PIÙ INFLUENTI (Consenso Globale) 🧬"
            print(f"\n{titolo}")
            f.write(f"{titolo}\n===========================================\n")
            for i, row in df_fi_globale.head(10).iterrows():
                testo = f"{i+1}. {row['Feature']} (Presente {row['Frequenza_nella_Top_15']} volte - Rank medio: {row['Rank_Medio']:.1f})"
                print(testo)
                f.write(f"{testo}\n")
            f.write("\n")

    # =========================================================================
    # GENERAZIONE GRAFICI AVANZATI
    # =========================================================================
    sns.set_theme(style="whitegrid")

    # 1. NUOVO GRAFICO: Barplot per le 3 CLASSI (Top 3 Bilanciati)
    if not df_bilanciati.empty:
        top_3 = df_bilanciati.head(3)
        plot_data = []
        for _, row in top_3.iterrows():
            nome_display = f"{row['Modello_Label']}\n({row['Esperimento']})"
            if pd.notna(row['F_Class_0']):
                plot_data.append({"Modello": nome_display, "Classe": "0", "Score": row["F_Class_0"]})
                plot_data.append({"Modello": nome_display, "Classe": "1", "Score": row["F_Class_1"]})
                plot_data.append({"Modello": nome_display, "Classe": "2", "Score": row["F_Class_2"]})
        
        if plot_data:
            df_plot_class = pd.DataFrame(plot_data)
            plt.figure(figsize=(11, 6))
            sns.barplot(data=df_plot_class, x="Modello", y="Score", hue="Classe", palette="Set2")
            plt.title("Prestazioni per Classe (F-Score) - Top 3 Modelli Bilanciati")
            plt.ylabel("Score per singola Classe")
            plt.ylim(0, 1.05)
            plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_str, "plot_classi_Top3_Bilanciati.png"), dpi=300)
            plt.show()
            plt.close()

    # 2. Scatter Plot F1 vs F2
    if ha_f1 and ha_f2:
        df_scatter = df_perf.dropna(subset=['F1_Mean', 'F2_Mean'])
        if not df_scatter.empty:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df_scatter, x="F1_Mean", y="F2_Mean", hue="Modello_Base", s=150, alpha=0.8, palette="tab10")
            plt.axvline(df_scatter['F1_Mean'].median(), color='gray', linestyle='--', alpha=0.5, label='Mediana F1')
            plt.axhline(df_scatter['F2_Mean'].median(), color='gray', linestyle=':', alpha=0.5, label='Mediana F2')
            plt.title("Trade-off F1 vs F2 (Ottimo = Alto a Destra)")
            plt.xlabel("Macro-F1 (Precisione Globale)")
            plt.ylabel("Macro-F2 (Sensibilità Clinica)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_str, "plot_scatter_F1_vs_F2.png"), dpi=300)
            plt.close()

    # 3. Barplot F1
    if ha_f1:
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df_perf, x="Esperimento", y="F1_Mean", hue="Modello_Label", palette="tab20")
        plt.title("Confronto Macro-F1")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_str, "plot_barplot_F1.png"), dpi=300)
        plt.close()

    # 4. Barplot F2
    if ha_f2:
        df_f2_target = df_perf[df_perf['Metrica_Tuning'] == 'F2'].dropna(subset=['F2_Mean'])
        if not df_f2_target.empty:
            plt.figure(figsize=(14, 7))
            sns.barplot(data=df_f2_target, x="Esperimento", y="F2_Mean", hue="Modello_Label", palette="tab20_r")
            plt.title("Confronto Macro-F2 (Solo target clinici)")
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_str, "plot_barplot_F2.png"), dpi=300)
            plt.close()

    # 5. Grafico Feature Importance Globale
    if not df_fi.empty and not df_fi_globale.empty:
        plt.figure(figsize=(12, 8))
        top_plot = df_fi_globale.head(20)
        sns.barplot(data=top_plot, x="Frequenza_nella_Top_15", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title("Consenso Feature Importance (Presenza nella Top 15 di tutti i modelli)")
        plt.xlabel("Numero di volte in cui la feature è risultata determinante")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_str, "plot_feature_importanza_globale.png"), dpi=300)
        plt.close()

    print("\n" + "="*70)
    print(f"✅ Salvataggio completato in: {out_dir_str}")
    print("="*70 + "\n")

if __name__ == "__main__":
    df_prestazioni = costruisci_master_dataframe()
    df_importanza = raccogli_feature_importance()
    genera_report_e_grafici(df_prestazioni, df_importanza)