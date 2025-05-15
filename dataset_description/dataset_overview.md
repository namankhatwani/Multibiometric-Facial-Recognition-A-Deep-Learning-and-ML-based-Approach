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



