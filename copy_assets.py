import shutil
import os

src_dir = r"C:\Users\pavan\.gemini\antigravity\brain\20840889-b788-4202-9b2c-7924458cc924"
dest_dir = r"c:\Users\pavan\Downloads\IPL_WINNER-PREDICTION-main\frontend\public"

images = {
    "bowler_silhouette_1779082919941.png": "bowler.png",
    "batsman_silhouette_1779082937793.png": "batsman.png",
    "stadium_panoramic_1779083063403.png": "stadium.png"
}

for src_name, dest_name in images.items():
    src_path = os.path.join(src_dir, src_name)
    dest_path = os.path.join(dest_dir, dest_name)
    if os.path.exists(src_path):
        print(f"Copying {src_path} -> {dest_path}")
        shutil.copy(src_path, dest_path)
    else:
        print(f"Source not found: {src_path}")
