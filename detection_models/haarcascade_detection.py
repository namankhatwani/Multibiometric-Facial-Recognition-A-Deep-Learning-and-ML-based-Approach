import os
import cv2
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

img_dir = '/content/drive/MyDrive/diss_pic_alll'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

disguise_map = {
    'normal': ['no'],
    'easy': ['ex', 'sp', 'ca'],
    'medium': ['to', 'ma', 'ga', 'ts', 'cs', 'ms', 'cg'],
    'hard': ['cm', 'gt', 'gm', 'gz', 'sz']
}

modality_map = {'v': 'Visible', 'i': 'Infrared', 't': 'Thermal'}
mod_names = ['Visible', 'Infrared', 'Thermal']
categories = ['normal', 'easy', 'medium', 'hard']

difficulty_results = {m: defaultdict(lambda: {'total': 0, 'correct': 0}) for m in mod_names}
addon_results = defaultdict(lambda: {m: {'total': 0, 'correct': 0} for m in mod_names})

for fname in tqdm(os.listdir(img_dir)):
    if not fname.endswith(('.jpg', '.jpeg', '.png')):
        continue
    try:
        parts = fname.split('_')
        if len(parts) < 4:
            continue
        addon = parts[1]
        mod_code = parts[2]
        modality = modality_map.get(mod_code)
        if modality is None:
            continue
        category = None
        for cat, lst in disguise_map.items():
            if addon in lst:
                category = cat
                break
        if category is None:
            continue

        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        difficulty_results[modality][category]['total'] += 1
        addon_results[addon][modality]['total'] += 1
        if len(faces) > 0:
            difficulty_results[modality][category]['correct'] += 1
            addon_results[addon][modality]['correct'] += 1
    except:
        continue

diff_data = {}
for mod in mod_names:
    row = []
    for cat in categories:
        correct = difficulty_results[mod][cat]['correct']
        total = difficulty_results[mod][cat]['total']
        acc = round((correct / total) * 100, 2) if total > 0 else 0.0
        row.append(acc)
    avg = round(sum(row) / len(row), 2)
    row.append(avg)
    diff_data[mod] = row

df_difficulty = pd.DataFrame.from_dict(diff_data, orient='index', columns=[c.capitalize() for c in categories] + ['Average'])
df_difficulty.index.name = 'Modality'
print("\nDetection Accuracy by Difficulty (Haarcascade):")
print(df_difficulty)

all_addons = sum(disguise_map.values(), [])
addon_rows = []

for addon in sorted(all_addons):
    row = [addon]
    for mod in mod_names:
        correct = addon_results[addon][mod]['correct']
        total = addon_results[addon][mod]['total']
        acc = round((correct / total) * 100, 2) if total > 0 else 0.0
        row.append(acc)
    addon_rows.append(row)

df_addon = pd.DataFrame(addon_rows, columns=['Add-on'] + mod_names)
df_addon.set_index('Add-on', inplace=True)
print("\nDetection Accuracy by Add-on (Haarcascade):")
print(df_addon)
