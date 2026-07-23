# DTC Ticket Matcher

Matches each DTC (drop-ship / pass-through) order to its partner outbound ticket and,
where one exists, the inbound ticket(s) that supplied it — by weight, within a
same-day window, with a supplier-roster cross-check for confidence grading.

## Folder layout

```
daily_inputs/                              raw exports you drop in
  Inbound Milton 6-18-26 ytd.csv
  Inbound Merrilville 6-18-26 ytd.csv
  Outbound Milton 6-18-26 ytd.csv
  Outbound Merrilville 6-18-26 ytd.csv
  Copy of MASTER ____ BMR-MMR Dashboard v11 - DTC_raw_data.csv   raw DTC export
data/
  merge_orders.py                          combines the 2 inbound + 2 outbound files
  2026 ytd combined inbound.csv            output of merge_orders.py (last run: 2026-06-18)
  2026 ytd combined outbound.csv           output of merge_orders.py (last run: 2026-06-18)
dtc/
  dtc_weight_match.py                      the matcher (run this last)
  BMR_Transport_Data.csv                   supplier/material roster, validity filter
  dtc_weight_matches.csv                   full match results  (last run: 2026-06-18)
  dtc_match_exceptions.csv                 unresolved/flagged rows only
  dtc_ticket_match.csv                     chart data, plain-text twin of the PNG
  dtc_ticket_match.png                     color-coded match table (green=high conf, red=no match)
```

Everything the code needs is inside this folder — both scripts resolve their input/output
paths relative to their own location (or the folder root), never back into the
original project. Verified by running both scripts from a copy of this folder alone.

## Inputs

You do **not** provide combined files. Each location exports its own inbound and
outbound tickets, so the real inputs are four raw yard files:

- `Inbound Milton ... ytd.csv`
- `Inbound Merrilville ... ytd.csv`
- `Outbound Milton ... ytd.csv`
- `Outbound Merrilville ... ytd.csv`

plus the raw DTC order export. `data/merge_orders.py` automatically stacks the two
inbound files and the two outbound files together (sorted by date, row-numbered) and
writes the result into `data/2026 ytd combined inbound.csv` / `2026 ytd combined
outbound.csv` — you never hand-build those.

## Running it

```
pip install -r requirements.txt
python data/merge_orders.py        # combines the 4 raw yard files -> data/
python dtc/dtc_weight_match.py     # runs the match, writes outputs into dtc/
```

Both scripts are run from this folder's root (not from inside `data/` or `dtc/`).

## Note on the DTC raw file

`dtc/dtc_weight_match.py` expects the raw DTC file to have a `DTC Row Id` column as
its first field. The source project's most recent export had dropped that column
(and used a different filename), so for this deliverable it was regenerated here:
`DTC Row Id` was added back as a simple sequential row number (1..85, same pattern
`data/merge_orders.py` uses for `Inbound Row Id` / `Outbound Row Id`), and the file
was saved under the name the script expects
(`daily_inputs/Copy of MASTER ____ BMR-MMR Dashboard v11 - DTC_raw_data.csv`). No
other data was changed. When you get a fresh DTC export going forward, drop it in
under that same filename with a `DTC Row Id` first column (or update the filename in
`main()`).

## What the matcher does

1. For each DTC order, find its linked outbound ticket by ID and confirm the net
   weight matches.
2. Search inbound tickets at the same yard/material within a 1-day window for a net
   weight match — first an exact single-ticket match, then a combination of tickets
   summing to the target (up to 6 tickets), then a widened search outside the
   same-day window as a last resort.
3. Check whether the inbound supplier and the DTC consumer are the same party
   (a red flag for a genuine pass-through) and grade confidence High/Medium/Low
   accordingly.
4. Cross-check the order's material against `BMR_Transport_Data.csv` — a roster of
   material/supplier combinations that actually move as DTC — to drop non-DTC
   materials and to suggest which suppliers to chase for genuinely missing legs.
