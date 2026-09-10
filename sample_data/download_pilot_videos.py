"""One-off script: downloads Godrej's real pilot videos (from the brief's
Drive link) into sample_data/godrej_pilot/. Run once, not part of the pipeline."""
import gdown
import os

FILES = {
    "1lxxxOIHBDP9lR9WG4iuQJkFWsN5uR9k3": "dock_level_dragging_cupboard.mp4",
    "11fq_efFs-V8rdtaQ6iEOHA1oRRel4uMC": "kd_packets_dragged_heavy_box_on_other_packets.mp4",
    "1MwUITPf0E58iIBN-MfbRn4aqJfCmAA1p": "rolling_and_dragging_on_wet_floor.mp4",
    "1erCepnU2FxMCYXBcaqp1rQcOKTlARkCO": "rolling_and_dropping_carton.mp4",
    "1L-jDeEAyal0aosWAvUu2pFpJPyPPTcqJ": "stepping_on_cartons_vertical_product_horizontal_heavy_on_top.mp4",
    "1jiBPIdGeowSH3SO9vNgsmHs7k7Zy91B9": "throwing_mattresses.mp4",
    "1LLQ69OL7I2mtI7ktmpNtt-9nNvuMVpcp": "throwing_seating_cartons_using_strap_to_hold.mp4",
}

out_dir = os.path.join(os.path.dirname(__file__), "godrej_pilot")
os.makedirs(out_dir, exist_ok=True)

for file_id, filename in FILES.items():
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        print(f"skip (exists): {filename}")
        continue
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"downloading {filename} ...")
    gdown.download(url, out_path, quiet=False)
