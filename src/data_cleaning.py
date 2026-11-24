"""
data_cleaning.py
=================
Ingests the three raw "systems of record" workbooks (synthetic stand-ins for
real membership-office spreadsheets) and produces a single, clean,
deduplicated member-level dataset.

This mirrors a very common real-world data engineering problem: the same
person exists in multiple spreadsheets maintained by different people, with
inconsistent identifiers (ASU ID is sometimes an int, sometimes the literal
string "Unknown"), inconsistent zip formats (5-digit vs ZIP+4), and no
single reliable foreign key to join on. We build a robust linkage key from
normalized name + zip and use it to merge records across sources.

Run directly to produce data/processed/members_clean.csv:
    python -m src.data_cleaning
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

REGULAR_MEMBERS_FILE = RAW_DIR / "Regular_Members_and_Historical_List_SYNTHETIC.xlsx"
DUES_FILE = RAW_DIR / "Membership_Dues_-_EV_Subscriptions_SYNTHETIC.xlsx"
EV_CURRENT_FILE = RAW_DIR / "EV_Subscriptions_-_CURRENT_SYNTHETIC.xlsx"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def normalize_text(value) -> str:
    """Lowercase, strip accents/punctuation/whitespace for fuzzy matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def normalize_zip(value) -> str:
    """Collapse ZIP+4 ('85287-1234') and numeric/float zips to a bare 5-digit string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    text = text.split("-")[0].split(".")[0]
    text = re.sub(r"[^0-9]", "", text)
    return text.zfill(5)[:5] if text else ""


def normalize_asu_id(value):
    """Return a clean int ASU ID, or None if missing/unusable ('Unknown', blank, NaN)."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or not text.isdigit():
        return None
    return int(text)


def linkage_key(first_name, last_name, zip_code) -> str:
    """Composite fuzzy-match key: normalized last+first name + 5-digit zip."""
    return f"{normalize_text(last_name)}|{normalize_text(first_name)}|{normalize_zip(zip_code)}"


def is_status_note_row(row, note_cols) -> bool:
    """Detect rows where a free-text administrative note was recorded instead of real data
    (e.g. 'Has requested removal from all email lists' sitting in the Discipline column)."""
    for col in note_cols:
        val = row.get(col)
        if isinstance(val, str) and "removal" in val.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Loaders — one function per raw sheet, each returns a tidy DataFrame
# ---------------------------------------------------------------------------
def load_active_members() -> pd.DataFrame:
    df = pd.read_excel(REGULAR_MEMBERS_FILE, sheet_name="Active Regular Members")
    df.columns = [str(c).strip() for c in df.columns]
    flagged = df.apply(lambda r: is_status_note_row(r, ["Discipline", "Joined EC"]), axis=1)
    df["opted_out_of_email"] = flagged
    df.loc[flagged, ["Discipline", "Joined EC", "Retirement Date", "Updated"]] = np.nan
    df["asu_id"] = df["ASU ID"].apply(normalize_asu_id)
    df["source_status"] = "active"
    df = df.rename(columns={
        "First Name": "first_name", "Last Name": "last_name", "Address": "address",
        "City": "city", "State": "state", "Zip": "zip", "Phone": "phone", "Email": "email",
        "Discipline": "discipline", "Joined EC": "joined_ec", "Retirement Date": "retirement_date",
    })
    df["linkage_key"] = df.apply(lambda r: linkage_key(r["first_name"], r["last_name"], r["zip"]), axis=1)
    keep = ["linkage_key", "asu_id", "first_name", "last_name", "address", "city", "state", "zip",
            "phone", "email", "discipline", "joined_ec", "retirement_date",
            "source_status", "opted_out_of_email"]
    return df[keep]


def load_deceased_members() -> pd.DataFrame:
    df = pd.read_excel(REGULAR_MEMBERS_FILE, sheet_name="Deceased Regular Members")
    df.columns = [str(c).strip() for c in df.columns]
    df["asu_id"] = df["ASU ID"].apply(normalize_asu_id)
    df["source_status"] = "deceased"
    df = df.rename(columns={
        "First Name": "first_name", "Last Name": "last_name", "Address": "address",
        "City": "city", "State": "state", "Zip": "zip", "E-mail": "email",
        "Department": "discipline", "Deceased Date": "deceased_date",
    })
    df["linkage_key"] = df.apply(lambda r: linkage_key(r["first_name"], r["last_name"], r["zip"]), axis=1)
    keep = ["linkage_key", "asu_id", "first_name", "last_name", "address", "city", "state", "zip",
            "email", "discipline", "deceased_date", "source_status"]
    return df[keep]


def load_special_cases() -> pd.DataFrame:
    df = pd.read_excel(REGULAR_MEMBERS_FILE, sheet_name="Special Cases")
    df.columns = [str(c).strip() for c in df.columns]
    df["asu_id"] = df["ASU ID"].apply(normalize_asu_id)
    df["source_status"] = "special_case"
    df = df.rename(columns={
        "First Name": "first_name", "Last Name": "last_name", "Address": "address",
        "City": "city", "State": "state", "Zip": "zip", "Phone": "phone", "Email": "email",
    })
    df["linkage_key"] = df.apply(lambda r: linkage_key(r["first_name"], r["last_name"], r["zip"]), axis=1)
    keep = ["linkage_key", "asu_id", "first_name", "last_name", "address", "city", "state", "zip",
            "phone", "email", "source_status"]
    return df[keep]


def load_dues_and_ev() -> pd.DataFrame:
    """The '2026 -2027' tab carries the dues + EV subscription facts we need."""
    df = pd.read_excel(DUES_FILE, sheet_name="2026 -2027")
    df.columns = [str(c).strip() for c in df.columns]
    # The sheet has two columns literally named "Date Paid" (one for the
    # membership fee, one for the EV subscription) — disambiguate by position.
    cols = list(df.columns)
    date_paid_positions = [i for i, c in enumerate(cols) if c == "Date Paid"]
    if len(date_paid_positions) == 2:
        cols[date_paid_positions[0]] = "dues_date_paid"
        cols[date_paid_positions[1]] = "ev_subscription_date_paid"
        df.columns = cols
    df = df.rename(columns={
        "First Name": "first_name", "Last Name": "last_name", "Zip": "zip",
        "Membership Fee": "membership_fee", "EV Subscription": "ev_subscription",
    })
    df["linkage_key"] = df.apply(lambda r: linkage_key(r["first_name"], r["last_name"], r["zip"]), axis=1)
    keep = ["linkage_key", "membership_fee", "dues_date_paid", "ev_subscription", "ev_subscription_date_paid"]
    return df[[c for c in keep if c in df.columns]]


def load_ev_current() -> pd.DataFrame:
    """Current EV subscription roster — gives us a recency/engagement signal
    (actively renewing a paid print subscription) independent of dues payment."""
    df = pd.read_excel(EV_CURRENT_FILE, sheet_name="Current EV Subscriptions")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={
        "First Name": "first_name", "Last Name": "last_name", "Zip": "zip",
        "Amount Paid": "ev_amount_paid", "Status": "ev_status",
    })
    df["linkage_key"] = df.apply(lambda r: linkage_key(r["first_name"], r["last_name"], r["zip"]), axis=1)
    df["has_current_ev_subscription"] = True
    keep = ["linkage_key", "ev_amount_paid", "ev_status", "has_current_ev_subscription"]
    return df[keep]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_clean_dataset() -> pd.DataFrame:
    active = load_active_members()
    deceased = load_deceased_members()
    special = load_special_cases()
    dues = load_dues_and_ev()
    ev_current = load_ev_current()

    # Union the three "person roster" sources into one row-per-membership-record
    # table. A person can legitimately appear once; status tells us which roster
    # they came from. Active takes priority over special_case/deceased duplicates
    # that are really the same linkage key (data-entry artifacts in the source).
    roster = pd.concat([active, deceased, special], ignore_index=True, sort=False)

    status_priority = {"active": 0, "special_case": 1, "deceased": 2}
    roster["_priority"] = roster["source_status"].map(status_priority)
    roster = roster.sort_values(["linkage_key", "_priority"])
    before = len(roster)
    roster = roster.drop_duplicates(subset="linkage_key", keep="first").drop(columns="_priority")
    duplicates_removed = before - len(roster)

    # Left-join in dues/EV facts keyed on the same fuzzy linkage key
    merged = roster.merge(dues, on="linkage_key", how="left")
    merged = merged.merge(ev_current, on="linkage_key", how="left")
    merged["has_current_ev_subscription"] = merged["has_current_ev_subscription"].fillna(False)
    merged["opted_out_of_email"] = merged.get("opted_out_of_email", False).fillna(False)

    # Light type cleanup
    for col in ["joined_ec", "retirement_date", "deceased_date", "dues_date_paid", "ev_subscription_date_paid"]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], errors="coerce")

    merged["member_id"] = range(1, len(merged) + 1)
    merged.attrs["duplicates_removed"] = duplicates_removed

    front = ["member_id", "linkage_key", "asu_id", "first_name", "last_name"]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    return merged


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean = build_clean_dataset()
    out_path = PROCESSED_DIR / "members_clean.csv"
    clean.to_csv(out_path, index=False)
    print(f"Wrote {len(clean)} cleaned member records to {out_path}")
    print(f"Duplicate records removed during linkage: {clean.attrs.get('duplicates_removed', 0)}")
    print(f"Records with a resolved numeric ASU ID: {clean['asu_id'].notna().sum()} / {len(clean)}")
    print(f"Status breakdown:\n{clean['source_status'].value_counts()}")


if __name__ == "__main__":
    main()
