import os
import cv2
import pandas as pd
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from collections import defaultdict

img_folder = '/content/drive/MyDrive/diss_pic_alll'
detector = MTCNN()

difficulty_map = {
    'no': 'Normal',
    'ca': 'Easy', 'ex': 'Easy', 'sp': 'Easy',
    'to': 'Medium', 'ma': 'Medium', 'ga': 'Medium', 'ts': 'Medium', 'cs': 'Medium',
    'ms': 'Hard', 'cg': 'Hard', 'cm': 'Hard', 'gt': 'Hard',
    'gm': 'Hard', 'gz': 'Hard', 'sz': 'Hard'
}

modalities = {'v': 'Visible', 'i': 'Infrared', 't': 'Thermal'}
mod_names = ['Visible', 'Infrared', 'Thermal']
categories = ['Normal', 'Easy', 'Medium', 'Hard']

difficulty_results = {m: defaultdict(lambda: {'total': 0, 'correct': 0}) for m in mod_names}
addon_results = defaultdict(lambda: {m: {'total': 0, 'correct': 0} for m in mod_names})

for fname in tqdm(os.listdir(img_folder)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    try:
        parts = fname.split('_')
        if len(parts) < 4:
            continue
        addon = parts[1]
        modality_key = parts[2]
        modality = modalities.get(modality_key)
        category = difficulty_map.get(addon)

        if modality is None or category is None:
            continue

        img_path = os.path.join(img_folder, fname)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        difficulty_results[modality][category]['total'] += 1
        addon_results[addon][modality]['total'] += 1

        if faces:
            difficulty_results[modality][category]['correct'] += 1
            addon_results[addon][modality]['correct'] += 1
    except:
        continue

difficulty_data = {}
for mod in mod_names:
    row = []
    for cat in categories:
        correct = difficulty_results[mod][cat]['correct']
        total = difficulty_results[mod][cat]['total']
        acc = round((correct / total) * 100, 2) if total > 0 else 0.0
        row.append(acc)
    avg = round(sum(row) / len(row), 2)
    row.append(avg)
    difficulty_data[mod] = row

df_difficulty = pd.DataFrame.from_dict(difficulty_data, orient='index', columns=categories + ['Average'])
df_difficulty.index.name = "Modality"
print("\nDetection Accuracy by Difficulty:")
print(df_difficulty)

addon_rows = []
for addon in sorted(addon_results.keys()):
    row = [addon]
    for mod in mod_names:
        correct = addon_results[addon][mod]['correct']
        total = addon_results[addon][mod]['total']
        acc = round((correct / total) * 100, 2) if total > 0 else 0.0
        row.append(acc)
    addon_rows.append(row)

df_addon = pd.DataFrame(addon_rows, columns=["Add-on"] + mod_names)
df_addon.set_index("Add-on", inplace=True)
print("\nDetection Accuracy by Add-on:")
print(df_addon)
