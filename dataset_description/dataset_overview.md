# Dataset Overview: Multibiometric Facial Dataset

This dataset was created as part of the dissertation titled **"Multibiometric Facial Retrieval and Recognition using Traditional ML and Deep Learning Approaches"**. It includes facial images of 22 Indian subjects captured under varied occlusion conditions, poses, and sensing modalities.

---

## 📸 Subjects and Image Count

- **Total Subjects**: 22
- **Visible Images per Subject**: 80  
- **Thermal/Infrared Images per Subject**: 42 each (only for 7 selected subjects)
- **Total Images**: 1800+ visible + 500+ thermal/infrared

---

## 🧩 Naming Convention

Each image follows the naming format:
<subject_id><addon><modality>_<pose>.jpg


- `subject_id`: `01` to `22`
- `addon`: facial occlusion code (e.g., `no`, `ca`, `gz`)
- `modality`:  
  - `v` = Visible  
  - `t` = Thermal  
  - `i` = Infrared
- `pose`:  
  - `f` = front  
  - `u` = up  
  - `d` = down  
  - `l` = left  
  - `r` = right

**Example**: `04_ma_v_d.jpg`  
→ Subject 04, wearing mask, visible image, looking down

---

## 🧢 Facial Add-on Categories

Grouped by difficulty:

### 🔹 Normal
- `no`: No occlusion

### 🟢 Easy
- `ex`: Eyeglasses  
- `sp`: Sunglasses  
- `ca`: Cap

### 🟠 Medium
- `to`: Towel  
- `ma`: Mask  
- `ga`: Goggles  
- `ts`: Turban & Sunglasses  
- `cs`: Cap & Sunglasses  
- `ms`: Mask & Sunglasses  
- `cg`: Cap & Goggles

### 🔴 Hard
- `cm`: Cap & Mask  
- `gt`: Goggles & Towel  
- `gm`: Goggles & Mask  
- `gz`: Goggles & Shawl  
- `sz`: Sunglasses & Shawl

---

## 🔥 Modalities

- **Visible (`v`)**: Full dataset (22 subjects × 80 images)
- **Thermal (`t`)** and **Infrared (`i`)**:  
  Available for subjects: `01, 04, 06, 08, 11, 12, 13`  
  Each subject has 42 thermal and 42 infrared images in front pose

---

## 🔁 Pose Details (for visible modality)

For each add-on combination:
- 5 head poses: `f` (front), `u` (up), `d` (down), `l` (left), `r` (right)

Thermal and IR only include `f` (front) pose.

---

## 🔒 Notes

- All images are stored in:  
  `/content/drive/MyDrive/diss_pic_alll`
- Format: `.jpg`
- Dataset not publicly released; used strictly for research and evaluation

---

# Notes on Modalities: Visible vs Thermal and Infrared

In the constructed multibiometric facial recognition dataset, three sensing modalities are used:

- **Visible (`v`)**
- **Thermal (`t`)**
- **Infrared (`i`)**

---

## 📊 Image Distribution Overview

| Modality   | Subjects Covered | Add-on Combinations | Poses        | Images per Subject | Total Images |
|------------|------------------|----------------------|--------------|---------------------|--------------|
| Visible    | 22               | 16                   | 5 poses (f,u,d,l,r) | 80                  | 1760         |
| Thermal    | 7 (01,04,06,08,11,12,13) | 6 (`no`, `to`, `ga`, `gm`, `gt`, `gz`) | Only `f` (front)  | 6            | 42           |
| Infrared   | 7 (same as thermal)     | 6 (same)              | Only `f` (front)  | 6            | 42           |

---

## 📉 Why Are Thermal and Infrared Image Counts Lower?

1. **Hardware Limitation**  
   Thermal and infrared image acquisition required specialized sensors, which were available only during limited lab sessions.

2. **Subject Availability**  
   Only 7 out of 22 subjects were available during sessions when thermal/infrared cameras were set up and working properly.

3. **Pose and Add-on Restrictions**  
   Due to time and sensor constraints:
   - Only **front pose (`f`)** was captured
   - Only a **subset of 6 add-ons** (including masks, towels, goggles) were used

---

## ⚠️ Implications for Experiments

- Models trained on **visible images** cannot be directly tested on thermal/infrared without modality adaptation.
- Evaluation on thermal/infrared is **limited to verification** tasks within the same modality (or cross-modal scenarios using subset matching).
- Results involving thermal and IR should always be **interpreted with the understanding of data imbalance**.

---

## ✅ Recommendations for Use

- Use **visible modality** for baseline and full pose/add-on evaluation.
- Use **thermal/infrared** for:
  - Cross-modal robustness testing
  - Low-light scenario simulation
  - Occlusion-insensitive recognition tasks
- Avoid combining all three modalities for unified training unless proper normalization and augmentation are applied.

---

## 📝 Summary

The dataset prioritizes **realistic disguise scenarios** in the visible domain and supplements it with **limited thermal/IR data** to demonstrate the potential of multibiometric recognition. Although the thermal/infrared coverage is partial, it adds significant value for exploring robustness in extreme sensing conditions.


