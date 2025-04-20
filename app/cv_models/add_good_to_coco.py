import os, json

# --- CONFIGURE THESE PATHS ---
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "train")
IMAGES_DIR = SPLIT_DIR
JSON_IN    = next(f for f in os.listdir(SPLIT_DIR) if f.endswith(".json"))
JSON_PATH  = os.path.join(SPLIT_DIR, JSON_IN)
# -----------------------------

# 1) Load existing COCO JSON
coco = json.load(open(JSON_PATH, "r"))

# 2) Determine next available image_id
existing_ids = {img["id"] for img in coco["images"]}
next_id = max(existing_ids) + 1

# 3) For every file in images/ that’s NOT already in coco["images"], add it
existing_files = {img["file_name"] for img in coco["images"]}
for fname in os.listdir(IMAGES_DIR):
    if fname in existing_files:
        continue
    # optional: filter extensions if you want .jpg/.png only
    new_entry = {
        "id": next_id,
        "file_name": fname
        # you can add "width" and "height" if you like, but our loader only needs id & file_name
    }
    coco["images"].append(new_entry)
    next_id += 1

# 4) Save back to JSON (you might backup the original first!)
backup = JSON_PATH + ".bak"
os.rename(JSON_PATH, backup)
with open(JSON_PATH, "w") as f:
    json.dump(coco, f, indent=2)
print(f"Added good images to {JSON_PATH} (backup at {backup})")
