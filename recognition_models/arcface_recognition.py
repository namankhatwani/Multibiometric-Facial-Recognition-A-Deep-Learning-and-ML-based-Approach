import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from collections import defaultdict
import random

DATASET_PATH = '/content/drive/MyDrive/diss_pic_alll'

addon_categories = {
    'Normal': ['no'],
    'Easy': ['ex', 'sp', 'ca'],
    'Medium': ['to', 'ma', 'ga', 'ts', 'cs', 'ms', 'cg'],
    'Hard': ['cm', 'gt', 'gm', 'gz', 'sz']
}

modality_map = {'v': 'Visible', 'i': 'Infrared', 't': 'Thermal'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
arcface_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
arcface_app.prepare(ctx_id=0)

train_set = defaultdict(list)
for img_name in os.listdir(DATASET_PATH):
    if not img_name.endswith(('.jpg', '.png')):
        continue
    parts = img_name.split('_')
    if len(parts) < 4:
        continue
    sid, addon, modality_key = parts[0], parts[1], parts[2]
    modality = modality_map.get(modality_key)
    category = next((k for k, v in addon_categories.items() if addon in v), None)
    if not modality or not category:
        continue
    key = (sid, addon, modality, category)
    train_set[key].append(img_name)

gallery = {}
query = []
for key, imgs in train_set.items():
    random.shuffle(imgs)
    gallery[key] = imgs[0]
    query += [(key[0], img) for img in imgs[1:]]

def extract_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img)[:, :, ::-1]
    faces = arcface_app.get(arr)
    if faces:
        return faces[0].embedding.reshape(1, -1)
    return None

embeddings_gallery = {}
for key, fname in tqdm(gallery.items(), desc="Gallery"):
    sid, addon, modality, category = key
    emb = extract_embedding(os.path.join(DATASET_PATH, fname))
    if emb is not None:
        embeddings_gallery[key] = (sid, emb)

difficulty_counts = {m: {c: [0, 0] for c in addon_categories} for m in modality_map.values()}
addon_list = sum(addon_categories.values(), [])
addon_counts = {m: {a: [0, 0] for a in addon_list} for m in modality_map.values()}

for sid_true, fname in tqdm(query, desc="Query"):
    parts = fname.split('_')
    addon, modality_key = parts[1], parts[2]
    modality = modality_map.get(modality_key)
    category = next((k for k, v in addon_categories.items() if addon in v), None)
    if not modality or not category:
        continue

    emb_query = extract_embedding(os.path.join(DATASET_PATH, fname))
    if emb_query is None:
        continue

    max_sim = -1
    pred_sid = None
    for (sid_ref, addon_ref, mod_ref, cat_ref), (sid_gal, emb_gal) in embeddings_gallery.items():
        if mod_ref != modality or cat_ref != category:
            continue
        sim = cosine_similarity(emb_query, emb_gal)[0][0]
        if sim > max_sim:
            max_sim = sim
            pred_sid = sid_gal

    difficulty_counts[modality][category][1] += 1
    addon_counts[modality][addon][1] += 1
    if pred_sid == sid_true:
        difficulty_counts[modality][category][0] += 1
        addon_counts[modality][addon][0] += 1

rows = []
for modality in modality_map.values():
    row = {'Modality': modality, 'Algorithm': 'ArcFace'}
    accs = []
    for cat in addon_categories:
        correct, total = difficulty_counts[modality][cat]
        acc = (correct / total) * 100 if total > 0 else 0
        row[cat] = round(acc, 2)
        accs.append(acc)
    row['Categorical average'] = round(np.mean(accs), 2)
    rows.append(row)

df_difficulty = pd.DataFrame(rows)
print("\nArcFace - Accuracy by Difficulty:")
print(df_difficulty)

rows = []
for addon in addon_list:
    row = [addon]
    for modality in modality_map.values():
        correct, total = addon_counts[modality][addon]
        acc = (correct / total) * 100 if total > 0 else 0
        row.append(round(acc, 2))
    rows.append(row)

df_addon = pd.DataFrame(rows, columns=["Add-on", "Visible", "Infrared", "Thermal"]).set_index("Add-on")
print("\nArcFace - Accuracy by Add-on:")
print(df_addon)
