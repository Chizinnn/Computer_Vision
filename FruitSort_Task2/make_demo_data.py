"""
Tải ảnh thật Apple, Banana, Orange từ Kaggle Fruits-360.
Nguồn: https://www.kaggle.com/datasets/moltean/fruits

Yêu cầu xác thực Kaggle trước khi chạy (xem README).
Ảnh được gom vào data/raw/<class>/ để tương thích prepare_dataset.py.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import re
import shutil

import kagglehub
from PIL import Image, ImageOps

DATASET = "moltean/fruits"
RAW = Path("data/raw")
FRUITS = ("apple", "banana", "orange")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalise_group_id(filename: str) -> str:
    """Gom các ảnh xoay của cùng một vật (ví dụ 31, r_31, r2_31)."""
    stem = Path(filename).stem.lower()
    stem = re.sub(r"^r\d*_", "", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_fruit_images(dataset_root: Path):
    """Tìm các thư mục lớp trong Training/Test, bỏ qua các lớp quả khác."""
    for folder in dataset_root.rglob("*"):
        if not folder.is_dir():
            continue
        parent_name = folder.parent.name.lower()
        if parent_name not in {"training", "train", "test", "testing"}:
            continue

        folder_name = folder.name.lower()
        fruit = next(
            (name for name in FRUITS
             if folder_name == name or folder_name.startswith(name + " ")),
            None,
        )
        if fruit:
            for image_path in folder.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield fruit, image_path


def main():
    print("Đang tải dataset Fruits-360 từ Kaggle...")
    try:
        dataset_path = Path(kagglehub.dataset_download(DATASET))
    except Exception as exc:
        raise SystemExit(
            "\nKhông tải được dataset từ Kaggle.\n"
            "Hãy đăng nhập/xác thực Kaggle trước khi chạy.\n"
            "Xem hướng dẫn trong README.md.\n"
            f"Chi tiết: {exc}"
        )

    found = list(find_fruit_images(dataset_path))
    if not found:
        raise SystemExit(
            f"Không tìm thấy thư mục Apple/Banana/Orange trong dataset: {dataset_path}"
        )

    # Xóa dữ liệu raw cũ để không trộn ảnh demo hoặc dataset khác.
    if RAW.exists():
        shutil.rmtree(RAW)
    for fruit in FRUITS:
        (RAW / fruit).mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    group_views = defaultdict(int)
    counts = defaultdict(int)
    rejected = 0

    for fruit, source in found:
        try:
            digest = sha256(source)
            if digest in seen_hashes:
                continue
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if image.width < 32 or image.height < 32:
                    rejected += 1
                    continue
                image.load()

                group_id = f"{fruit}_{normalise_group_id(source.name)}"
                group_views[group_id] += 1
                view_id = group_views[group_id]
                target = RAW / fruit / f"{group_id}__view{view_id}.jpg"
                image.save(target, format="JPEG", quality=95, optimize=True)

            seen_hashes.add(digest)
            counts[fruit] += 1
        except Exception as exc:
            rejected += 1
            print(f"Bỏ qua ảnh lỗi: {source} ({exc})")

    print("\nĐã tải và chuẩn bị dữ liệu Kaggle:")
    total = 0
    for fruit in FRUITS:
        print(f"  {fruit}: {counts[fruit]} ảnh")
        total += counts[fruit]

    if any(counts[fruit] == 0 for fruit in FRUITS):
        raise SystemExit("Thiếu ảnh ở ít nhất một lớp; hãy kiểm tra dataset tải về.")

    print(f"Tổng cộng: {total} ảnh; bỏ qua {rejected} ảnh lỗi/không hợp lệ.")
    print(f"Thư mục dữ liệu: {RAW.resolve()}")
    print("\nBước tiếp theo:")
    print("  python prepare_dataset.py")
    print("  python train.py --epochs 10 --batch-size 32 --lr 0.001")


if __name__ == "__main__":
    main()
