import os
import cv2
import numpy as np
import json
from paddleocr import PaddleOCR

# ===============================
# 1. TIỀN XỬ LÝ ẢNH (KHÔNG GHI FILE)
# ===============================
def preprocess_image(input_path, min_size=1600):
    img = cv2.imread(input_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if max(h, w) < min_size:
        scale = min_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    clahe = cv2.createCLAHE(3.0, (6,6))
    enhanced = clahe.apply(blur)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ===============================
# 2. NGƯỠNG TỰ ĐỘNG
# ===============================
def auto_x_threshold(rects):
    x_centers = sorted([(r[0]+r[2])/2 for r in rects])
    diffs = [x_centers[i]-x_centers[i-1] for i in range(1,len(x_centers))]
    if not diffs:
        return 30
    median_gap = np.median([d for d in diffs if d < np.percentile(diffs,10)])
    return max(15, median_gap*1.5)

def auto_y_threshold(rects):
    heights = [r[3]-r[1] for r in rects]
    if not heights:
        return 30
    return max(15, np.median(heights)*0.4)

# ===============================
# 3. OCR INIT
# ===============================
ocr = PaddleOCR(
    lang="ch",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# ===============================
# 4. XỬ LÝ 1 ẢNH (GIỮ NGUYÊN LOGIC)
# ===============================
def process_single_image(image_path, output_dir):
    prep_img = preprocess_image(image_path)
    if prep_img is None:
        return

    res = ocr.predict(input=prep_img)[0]
    dt_polys = res["dt_polys"]
    rec_texts = res["rec_texts"]

    if not dt_polys:
        return

    # ---- poly → rect
    def poly_to_rect(poly):
        arr = np.array(poly)
        return [
            int(arr[:,0].min()),
            int(arr[:,1].min()),
            int(arr[:,0].max()),
            int(arr[:,1].max())
        ]

    rects = [poly_to_rect(p) for p in dt_polys]

    # ---- lọc Y-centroid
    ys = np.array([(r[1]+r[3])/2 for r in rects])
    center_y = np.mean(ys)
    threshold = 0.25 * prep_img.shape[0]
    keep = np.abs(ys - center_y) <= threshold

    rects = [r for r,k in zip(rects,keep) if k]
    rec_texts = [t for t,k in zip(rec_texts,keep) if k]

    # ---- gộp cột
    def merge_columns(rects, texts, x_threshold, y_threshold):
        columns=[]
        for rect,txt in zip(rects,texts):
            x1,y1,x2,y2 = rect
            xc=(x1+x2)/2
            matched=False
            for col in columns:
                bx1,by1,bx2,by2 = col[0]["rect"]
                bxc=(bx1+bx2)/2
                if abs(xc-bxc)<x_threshold and (
                    abs(y1-by1)<y_threshold or abs(y2-by2)<y_threshold
                ):
                    col.append({"rect":rect,"text":txt})
                    matched=True
                    break
            if not matched:
                columns.append([{"rect":rect,"text":txt}])

        for col in columns:
            col.sort(key=lambda b:b["rect"][1])
        columns.sort(key=lambda c:-np.mean([(b["rect"][0]+b["rect"][2])/2 for b in c]))
        return columns

    x_th = min(20, auto_x_threshold(rects))
    y_th = auto_y_threshold(rects)
    columns = merge_columns(rects, rec_texts, x_th, y_th)

    # ---- gộp cột → box lớn
    merged_rects, merged_texts = [], []
    for col in columns:
        xs, ys, text = [], [], ""
        for b in col:
            x1,y1,x2,y2 = b["rect"]
            xs += [x1,x2]
            ys += [y1,y2]
            text += b["text"]
        merged_rects.append([min(xs),min(ys),max(xs),max(ys)])
        merged_texts.append(text)

    # ---- tạo thẻ
    def group_boxes_into_tags(rects, texts, x_threshold=50):
        boxes=[{"rect":r,"text":t,"xc":(r[0]+r[2])/2} for r,t in zip(rects,texts)]
        tags=[]
        for b in boxes:
            for tag in tags:
                if any(abs(b["xc"]-tb["xc"])<=x_threshold for tb in tag):
                    tag.append(b)
                    break
            else:
                tags.append([b])
        return tags

    def sort_boxes_in_tag(tag):
        tag.sort(key=lambda b:b["rect"][1])
        out=[]
        while tag:
            y=tag[0]["rect"][1]
            same=[b for b in tag if abs(b["rect"][1]-y)<10]
            same.sort(key=lambda b:-b["xc"])
            out+=same
            tag=[b for b in tag if b not in same]
        return out

    tags = [
        sort_boxes_in_tag(t)
        for t in sorted(
            group_boxes_into_tags(merged_rects, merged_texts),
            key=lambda t:-np.mean([b["xc"] for b in t])
        )
    ]

    # ===============================
    # 5. LƯU JSON CUỐI
    # ===============================
    base = os.path.splitext(os.path.basename(image_path))[0]

    final_json = {
        "input_path": image_path,
        "tags": [
            {
                "tag_id": i+1,
                "text": "".join(b["text"] for b in tag),
                "boxes": [
                    {"rect": b["rect"], "text": b["text"]}
                    for b in tag
                ]
            }
            for i, tag in enumerate(tags)
        ]
    }

    json_path = os.path.join(output_dir, base + ".json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    # ===============================
    # 6. VẼ ẢNH CUỐI
    # ===============================
    img_final = prep_img.copy()
    for i, tag in enumerate(tags):
        for b in tag:
            x1,y1,x2,y2 = b["rect"]
            cv2.rectangle(img_final,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(img_final,str(i+1),(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    img_path = os.path.join(output_dir, base + ".jpg")
    cv2.imwrite(img_path, img_final)

    print(f"[DONE] {base}")

# ===============================
# 7. CHẠY TOÀN BỘ FOLDER LỒNG NHAU
# ===============================
# INPUT_ROOT = "../Data/input/HVNH_37-41"
# OUTPUT_ROOT = "../Data/output/HVNH_37-41"

INPUT_ROOT = "../Data/input/HVNH_10-12"
OUTPUT_ROOT = "../Data/output/HVNH_10-12"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for subfolder in sorted(os.listdir(INPUT_ROOT)):
    input_subdir = os.path.join(INPUT_ROOT, subfolder)

    # chỉ xử lý folder
    if not os.path.isdir(input_subdir):
        continue

    output_subdir = os.path.join(OUTPUT_ROOT, subfolder)
    os.makedirs(output_subdir, exist_ok=True)

    print(f"\n[PROCESS FOLDER] {subfolder}")

    for fname in sorted(os.listdir(input_subdir)):
        if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif")):
            image_path = os.path.join(input_subdir, fname)
            process_single_image(image_path, output_subdir)
