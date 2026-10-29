"""
EICV7 Poverty Profile — Table Extractor
Converts the raw Excel file into clean JSON files for the Imari Health dashboard.

Source: NISR EICV7 Poverty Profile 2023/24
Reference: National Institute of Statistics of Rwanda (NISR) (2025)
"""

import json
import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "raw/eicv7_poverty_tables.xlsx"
OUTPUT_DIR = "processed"

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping: Excel sheet name → clean JSON output filename
SHEETS_TO_EXTRACT = {
    "Table 0.1. ": "poverty_by_province",
    "Table 0.2.":  "access_services",
    "Table 5.1.":  "poverty_by_area_province",
    "Table 5.2.":  "poverty_gap",
    "Table 5.4. ": "poverty_contribution",
    "Table 6.1.":  "poverty_by_hh_head",
    "Table 6.2.":  "poverty_by_gender_age",
    "Table 6.3.":  "poverty_by_education",
    "Table 6.6. ": "health_by_poverty",
    "Table 6.7. ": "housing_by_poverty",
    "Table 6.8. ": "durables_by_poverty",
    "Table 6.9.":  "savings_gap",
    "Table 6.10.": "shocks_by_poverty",
    "Table 6.11.": "coping_by_poverty",
    "Table 7.1. ": "inequality",
}


def clean_dataframe(df):
    """Convert DataFrame to JSON-safe records."""
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df.to_dict(orient="records")


def extract_sheet(xls, sheet_name):
    """Read a sheet, skipping the title row, and return JSON-safe records."""
    try:
        # Read with header at row 1 (skip title row 0)
        df = pd.read_excel(xls, sheet_name=sheet_name, header=1)
        records = clean_dataframe(df)
        return records
    except Exception as e:
        print(f"  [WARN] Could not read '{sheet_name}': {e}")
        return []


def main():
    print("=" * 60)
    print("EICV7 Poverty Profile — Table Extractor")
    print("=" * 60)

    # Verify input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        print(f"Place 'eicv7_poverty_tables.xlsx' in the 'raw/' folder.")
        return

    print(f"\n[OK] Reading: {INPUT_FILE}")
    xls = pd.ExcelFile(INPUT_FILE)
    print(f"[OK] Sheets available: {len(xls.sheet_names)}")

    # Extract each mapped sheet
    extracted = 0
    for sheet_name, output_name in SHEETS_TO_EXTRACT.items():
        print(f"\n→ Extracting: {sheet_name.strip()}")
        records = extract_sheet(xls, sheet_name)

        if not records:
            print(f"  [SKIP] No data extracted.")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{output_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)

        print(f"  [OK] Saved {len(records)} rows → {output_path}")
        extracted += 1

    print("\n" + "=" * 60)
    print(f"[DONE] Extracted {extracted} tables to '{OUTPUT_DIR}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()