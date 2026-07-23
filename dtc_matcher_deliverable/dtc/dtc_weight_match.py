"""Match each DTC order to its partner inbound and outbound tickets by weight."""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WINDOW_DAYS = 1
MAX_COMBO = 6

CODE_EQUIV = {"FETURN": {"FETURN", "TURNSTEEL"}, "TURNSTEEL": {"FETURN", "TURNSTEEL"}}

# DTC material name -> BMR Transport roster material name. The deprecated "ZZ DO NOT
# USE" aluminum codes (breakage / turnings) are one aluminum family in the roster.
ROSTER_ALIAS = {"ZZ DO NOT USE ALUMINUM BREAKAGE": "ZZ DO NOT USE - ALUMINUM TURNINGS"}


# clean the commas and dollar signs out of a column and make it numeric
def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("$", ""), errors="coerce")


def _subset(items, target):
    """Return the 'best' subset (list of (ticket,wt,date)) summing to target, or None.
    Prefers fewest tickets, then the tightest date span (most likely a real bin)."""
    target = int(round(target))
    pool = [(t, int(round(w)), d) for t, w, d in items if 0 < round(w) <= target]

    best = None
    for r in range(2, min(MAX_COMBO, len(pool)) + 1):
        for combo in combinations(pool, r):
            if sum(w for _, w, _ in combo) == target:
                span = (max(d for *_, d in combo) - min(d for *_, d in combo)).days
                key = (r, span)
                if best is None or key < best[0]:
                    best = (key, combo)
        if best:
            break
    return list(best[1]) if best else None


def _load_transport_roster(path):
    """BMR Transport pivot -> (valid materials, valid supplier companies, material->companies).

    The pivot lists every (material, supplier company) combination that actually
    moves as DTC. A material or company absent from it cannot be a DTC order, so it
    doubles as a validity filter and, for genuine missing legs, names the supplier
    companies worth chasing in GreenSpark."""
    df = pd.read_csv(path, header=1, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    meta = {"Carrier Name", "Material Name", "Grand Total"}
    companies = [c for c in df.columns if c not in meta]
    df = df[df["Material Name"].notna() & (df["Material Name"].str.strip() != "")]
    df = df[df["Material Name"].str.strip() != "Grand Total"]
    df["_mat"] = df["Material Name"].str.strip().str.upper()
    mat_to_co = {}
    for _, row in df.iterrows():
        cos = {c for c in companies if pd.notna(row[c]) and str(row[c]).strip() not in ("", "0")}
        mat_to_co.setdefault(row["_mat"], set()).update(cos)
    return set(df["_mat"]), {c.strip().upper() for c in companies}, mat_to_co


# match every DTC order to its outbound ticket, then hunt down the matching inbound twin
def main():
    dtc = pd.read_csv(os.path.join(ROOT, "daily_inputs", "Copy of MASTER ____ BMR-MMR Dashboard v11 - DTC_raw_data.csv"), dtype=str)
    inb = pd.read_csv(os.path.join(ROOT, "data", "2026 ytd combined inbound.csv"), dtype=str)
    out = pd.read_csv(os.path.join(ROOT, "data", "2026 ytd combined outbound.csv"), dtype=str)
    valid_mats, valid_companies, mat_to_co = _load_transport_roster(
        os.path.join(HERE, "BMR_Transport_Data.csv"))

    dtc["net"] = _num(dtc["Yard Net Weight"])
    inb["net"] = _num(inb["Net Weight"])
    out["net"] = _num(out["Net Weight"])
    inb["date"] = pd.to_datetime(inb["Effective Date"].str[:10], errors="coerce")
    out["date"] = pd.to_datetime(out["Date In"].str[:10], errors="coerce")

    # Spreadsheet row number as opened in Excel: header is row 1, so the first
    # data row is row 2 -> position (0-based) + 2.
    inb["csv_row"] = inb.index + 2
    out["csv_row"] = out.index + 2

    # collapse inbound into one row per ticket, keeping its supplier and source rows
    itix = inb.groupby(["Material Code", "Location", "Ticket #"], as_index=False).agg(
        net=("net", "sum"), date=("date", "min"),
        supplier=("Customer Name", "first"), vendor_class=("Vendor Class", "first"),
        row_ids=("csv_row", lambda s: ";".join(s.astype(str))))
    itix["tid"] = itix["Location"] + "/" + itix["Ticket #"]
    sup = itix.set_index("tid")[["supplier", "vendor_class"]]
    rid_by_tid = itix.set_index("tid")["row_ids"].to_dict()

    # first pass, find each order's outbound ticket and set up its inbound search window
    recs = []
    for _, o in dtc.iterrows():
        W = o["net"]
        tkt = o["Outbound Ticket Id"]
        rec = dict(dtc_row=o["DTC Row Id"], outbound_ticket=tkt, out_csv_row="", material=o["Material Name"],
                   dtc_net=W, out_method="", out_match="",
                   mat_code="", yard="", ship_date="",
                   in_method="NONE", in_match="", in_csv_row="", in_total="", consumer=o["Customer Name"],
                   _W=W, _cand=None, _intid=None)

        ot = out[out["Outbound Ticket #"] == tkt]
        hit = ot[ot["net"] == W]
        if len(hit):
            r = hit.iloc[0]
            rec.update(out_method="exact net (linked ticket)", out_match=tkt,
                       out_csv_row=r["csv_row"], mat_code=r["Material Code"],
                       yard=r["Location"], ship_date=str(r["date"].date()) if pd.notna(r["date"]) else "")
        elif len(ot):
            rec.update(out_method="ticket found, net mismatch", out_match=tkt,
                       out_csv_row=ot.iloc[0]["csv_row"],
                       mat_code=ot.iloc[0]["Material Code"], yard=ot.iloc[0]["Location"],
                       ship_date=str(ot.iloc[0]["date"].date()) if pd.notna(ot.iloc[0]["date"]) else "")
        else:
            rec.update(out_method="ticket NOT in outbound file")

        mc, yd, sd = rec["mat_code"], rec["yard"], rec["ship_date"]
        if mc and yd and sd:
            ship = pd.Timestamp(sd)
            codes = CODE_EQUIV.get(mc, {mc})
            pool = itix[(itix["Material Code"].isin(codes)) & (itix["Location"] == yd)].copy()
            rec["_pool"] = pool
            rec["_cand"] = pool[(pool["date"] >= ship - pd.Timedelta(days=WINDOW_DAYS)) &
                                (pool["date"] <= ship + pd.Timedelta(days=WINDOW_DAYS))].copy()
            rec["_ship"] = ship
        recs.append(rec)

    # work oldest first so each inbound ticket only gets claimed once
    consumed = set()
    order = sorted(range(len(recs)),
                   key=lambda i: recs[i].get("_ship", pd.Timestamp.max))

    def avail(c):
        return c[~c["tid"].isin(consumed)] if c is not None else None

    # pass 1, claim an inbound ticket whose net matches exactly inside the day window
    for i in order:
        r = recs[i]; c = avail(r["_cand"])
        if c is None or not len(c):
            continue
        ex = c[c["net"].round() == round(r["_W"])]
        if len(ex):
            ex = ex.assign(_d=(ex["date"] - r["_ship"]).abs()).sort_values(["_d"])
            pick = ex.iloc[0]
            consumed.add(pick["tid"])
            r.update(in_method="exact net", in_match=f"#{pick['Ticket #']}", in_csv_row=pick["row_ids"],
                     in_total=round(pick["net"]), _intid=pick["tid"])

    # pass 2, try to hit the target weight as a sum of several smaller tickets
    for i in order:
        r = recs[i]
        if r["in_method"] != "NONE":
            continue
        c = avail(r["_cand"])
        if c is None or not len(c):
            continue
        s = _subset(list(zip(c["tid"], c["net"], c["date"])), r["_W"])
        if s:
            for tid, *_ in s:
                consumed.add(tid)
            r.update(in_method="sum of nets",
                     in_match=" + ".join(f"#{tid.split('/')[1]}({int(round(w))})" for tid, w, _ in s),
                     in_csv_row=";".join(rid_by_tid.get(tid, "") for tid, *_ in s),
                     in_total=sum(int(round(w)) for _, w, _ in s))

    # pass 3, last resort, widen past the same day window and note how far off it landed
    for i in order:
        r = recs[i]
        if r["in_method"] != "NONE" or r.get("_pool") is None:
            continue
        p = avail(r["_pool"])
        if p is None or not len(p):
            continue

        ex = p[p["net"].round() == round(r["_W"])]
        if len(ex):
            ex = ex.assign(_d=(ex["date"] - r["_ship"]).abs()).sort_values("_d")
            pick = ex.iloc[0]; consumed.add(pick["tid"])
            gap = int(np.busday_count(r["_ship"].date(), pick["date"].date()))
            r.update(in_method="exact net (outside same-day)", in_match=f"#{pick['Ticket #']}",
                     in_csv_row=pick["row_ids"], in_total=round(pick["net"]), _gap=gap, _intid=pick["tid"])

    # attach the inbound supplier and flag it when supplier and consumer are the same party
    for r in recs:
        r.update(supplier="", supplier_class="", supplier_check="", confidence="")
        tid = r.get("_intid")
        if tid is None or tid not in sup.index:
            continue
        row = sup.loc[tid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        r["supplier"] = row["supplier"]
        r["supplier_class"] = row["vendor_class"]
        same = str(row["supplier"]).strip().lower() == str(r["consumer"]).strip().lower()
        r["supplier_check"] = "SUSPECT: supplier==consumer" if same else "ok (distinct counterparties)"

    # grade every match high medium or low and record why it failed when it did
    no_twin = set(_no_inbound_twin_anywhere(dtc, inb, out))
    for r in recs:
        m = r["in_method"]
        supp_ok = r["supplier_check"].startswith("ok")
        if "outside same-day" in m and abs(r.get("_gap", 0)) > 2:
            r["confidence"] = "Medium" if supp_ok else "Low"
            r["exception_reason"] = f"inbound ticket {r.get('_gap', '?')} business days outside same-day (not stored = should be same day)"
        elif m != "NONE":
            r["confidence"] = "High" if supp_ok else "Low"
            r["exception_reason"] = "" if r["confidence"] == "High" else (
                "supplier == consumer (implausible for a pass-through)")
        elif r["out_method"] == "ticket NOT in outbound file":
            r["exception_reason"] = "outbound ticket missing from outbound file"
        elif r["dtc_row"] in no_twin:
            r["exception_reason"] = "no inbound ticket of this net anywhere (missing inbound leg / not a pass-through)"
        else:
            r["exception_reason"] = "net twin exists only under a different material/yard (coincidental)"

    # BMR Transport roster filter: a material absent from the roster cannot be a DTC
    for r in recs:
        mat = str(r["material"]).strip().upper()
        mat = ROSTER_ALIAS.get(mat, mat)
        in_roster = mat in valid_mats
        r["roster_ok"] = "Y" if in_roster else "N"
        r["chase_suppliers"] = ""
        if not in_roster:
            if r["in_method"] == "NONE":
                r["exception_reason"] = "material not in BMR Transport roster - cannot be a DTC order (drop)"
                r["confidence"] = "n/a (not DTC)"
        elif r["in_method"] == "NONE":
            r["chase_suppliers"] = "; ".join(sorted(mat_to_co.get(mat, set())))

    # flatten the records to a frame and write the full matches and the exceptions
    res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    out_cols = [c for c in res.columns if c != "confidence"]
    res[out_cols].to_csv(os.path.join(HERE, "dtc_weight_matches.csv"), index=False)
    exc = res[res["exception_reason"] != ""].copy()
    exc[out_cols].to_csv(os.path.join(HERE, "dtc_match_exceptions.csv"), index=False)

    # print the run summary, match counts and the missing legs still worth chasing
    n = len(res)
    print(f"DTC orders: {n}\n")
    print("OUTBOUND match:")
    print(res["out_method"].value_counts().to_string(), "\n")
    high = (res["confidence"] == "High").sum()
    matched = (res["in_method"] != "NONE").sum()
    print("CONFIDENCE (net match, supplier != consumer):")
    print(f"  High (net+supplier ok)            : {high}")
    print(f"  matched but flagged for review    : {matched - high}")
    print(f"  no inbound found                  : {n - matched}")
    print("\nException reasons:")
    print(exc["exception_reason"].value_counts().to_string())

    dropped = res["roster_ok"].eq("N").sum()
    chase = res[res["exception_reason"].str.startswith("no inbound ticket of this net")]
    print(f"\nBMR Transport roster filter:")
    print(f"  dropped (material not in roster, not a DTC order): {dropped}")
    print(f"  genuine missing inbound legs to chase (in roster): {len(chase)}")
    print("\nMissing inbound legs to chase (with candidate suppliers from roster):")
    print(chase[["dtc_row", "outbound_ticket", "material", "dtc_net", "yard",
                 "ship_date", "chase_suppliers"]].to_string(index=False))
    render_chart(res, os.path.join(HERE, "dtc_ticket_match.png"))
    write_chart_csv(res, os.path.join(HERE, "dtc_ticket_match.csv"))
    print(f"\n-> wrote dtc/dtc_weight_matches.csv, dtc/dtc_match_exceptions.csv, "
          f"dtc/dtc_ticket_match.png, dtc/dtc_ticket_match.csv")


# Columns shown in the chart PNG and its CSV twin (dtc_ticket_match.csv). Both are
# built from this one spec via _chart_table, so they always carry identical rows/cols.
CHART_COLS = [("dtc_row", "DTC\nRow"), ("material", "Material"), ("dtc_net", "DTC\nNet"),
              ("outbound_ticket", "OB\nTicket"), ("out_csv_row", "OB\nCSV Row"),
              ("yard", "Yard"), ("ship_date", "Ship\nDate"),
              ("in_match", "Inbound\nTicket(s)"), ("in_csv_row", "Inbound\nCSV Row"),
              ("supplier", "Inbound Supplier"),
              ("exception_reason", "Exception / Note")]


def _chart_table(res, truncate=False):
    """Display frame behind both the PNG and its CSV twin. For genuine missing legs
    (no inbound supplier) the roster's candidate suppliers fill the supplier cell."""
    disp = res[[c for c, _ in CHART_COLS]].copy()
    disp["supplier"] = disp["supplier"].fillna("")
    disp["exception_reason"] = disp["exception_reason"].fillna("")
    disp["in_match"] = disp["in_match"].fillna("")
    chase_mask = (res["in_method"] == "NONE") & (res["chase_suppliers"].fillna("") != "")
    disp.loc[chase_mask, "supplier"] = "chase: " + res.loc[chase_mask, "chase_suppliers"].fillna("")
    if truncate:  # cosmetic, PNG only — the CSV twin keeps full text
        disp["material"] = disp["material"].str.slice(0, 22)
        disp["in_match"] = disp["in_match"].str.slice(0, 22)
        disp["supplier"] = disp["supplier"].str.slice(0, 46)
    return disp


def write_chart_csv(res, path):
    """CSV twin of the chart PNG — identical columns and rows, untruncated text."""
    disp = _chart_table(res, truncate=False)
    disp.columns = [h.replace("\n", " ") for _, h in CHART_COLS]
    disp.to_csv(path, index=False)


def render_chart(res, path):
    """Color-coded table image: green = High (net+supplier), amber = flagged, red = none."""
    cols = CHART_COLS
    disp = _chart_table(res, truncate=True)

    n = len(disp)
    fig, ax = plt.subplots(figsize=(24, 0.32 * n + 1.8))
    ax.axis("off")
    high = int((res["confidence"] == "High").sum())
    matched = int((res["in_method"] != "NONE").sum())
    ax.set_title(f"DTC Orders — Inbound/Outbound Matches, net+supplier corroborated  "
                 f"(dark-green {high} High | light-green {matched - high} matched/flagged | red {n - matched} no inbound)",
                 fontsize=13, pad=12)

    tbl = ax.table(cellText=disp.values, colLabels=[h for _, h in cols],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    for i in range(n):
        if res["in_method"].iloc[i] == "NONE":
            color = "#f5b7b1"
        elif res["confidence"].iloc[i] == "High":
            color = "#a8dba0"
        else:
            color = "#d7f0cf"
        for j in range(len(cols)):
            tbl[(i + 1, j)].set_facecolor(color)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#d9e1f2")
        tbl[(0, j)].set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _no_inbound_twin_anywhere(dtc, inb, out):
    """DTC rows whose net equals no single inbound ticket net on ANY date, either yard."""
    itix = inb.groupby(["Location", "Ticket #"], as_index=False)["net"].sum()
    nets = set(itix["net"].round())
    return [o["DTC Row Id"] for _, o in dtc.iterrows() if round(o["net"]) not in nets]


if __name__ == "__main__":
    main()
