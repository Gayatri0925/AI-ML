import os
import shutil
import hashlib
import cv2
from datetime import datetime
import humanize
from tabulate import tabulate

FILE_CATEGORIES = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"],
    "Videos": [".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"],
    "Documents": [".pdf", ".docx", ".doc", ".txt", ".pptx", ".xlsx", ".csv"],
    "Music": [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"],
    "Archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
    "Programs": [".exe", ".msi", ".apk", ".bat", ".sh", ".deb", ".pkg"],
    "Code": [".py", ".java", ".c", ".cpp", ".html", ".css", ".js", ".php", ".json", ".xml"],
    "Others": []
}

def get_category(ext):
    for category, exts in FILE_CATEGORIES.items():
        if ext.lower() in exts:
            return category
    return "Others"

def file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def detect_blur(image_path, threshold=100.0):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        return variance < threshold
    except Exception:
        return False

def organize_and_clean(folder_path):
    if not os.path.exists(folder_path):
        print("❌ Folder not found.")
        return

    start = datetime.now()
    files_moved = 0
    file_hashes = {}
    duplicates = []
    blur_images = []
    folder_sizes = {}

    print("\n🔍 Scanning all files (including subfolders)...")

    # Walk through all subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Skip files already inside category folders
            if any(cat in file_path for cat in FILE_CATEGORIES.keys()):
                continue

            ext = os.path.splitext(file)[1]
            category = get_category(ext)
            category_folder = os.path.join(folder_path, category)
            os.makedirs(category_folder, exist_ok=True)

            # Duplicate check
            try:
                file_md5 = file_hash(file_path)
                if file_md5 in file_hashes:
                    duplicates.append(file_path)
                    continue
                else:
                    file_hashes[file_md5] = file_path
            except Exception:
                continue

            # Blur check
            if category == "Images" and detect_blur(file_path):
                blur_images.append(file_path)

            # Move file
            new_path = os.path.join(category_folder, file)
            counter = 1
            while os.path.exists(new_path):
                filename, extn = os.path.splitext(file)
                new_path = os.path.join(category_folder, f"{filename}({counter}){extn}")
                counter += 1

            shutil.move(file_path, new_path)
            files_moved += 1

            folder_sizes[category] = folder_sizes.get(category, 0) + os.path.getsize(new_path)

    end = datetime.now()
    duration = (end - start).total_seconds()

    print(f"\n✅ Organized {files_moved} files in {duration:.2f} seconds.\n")

    if folder_sizes:
        print("📊 Storage Summary:")
        table = [[cat, humanize.naturalsize(size)] for cat, size in folder_sizes.items()]
        print(tabulate(table, headers=["Category", "Size Used"], tablefmt="grid"))
    else:
        print("No files organized (maybe all were already sorted).")

    # Show duplicates
    if duplicates:
        print(f"\n⚠️ Found {len(duplicates)} duplicate files:")
        for dup in duplicates:
            print("  -", dup)
        delete = input("\nDo you want to delete all duplicates? (y/n): ").lower()
        if delete == "y":
            for dup in duplicates:
                try:
                    os.remove(dup)
                except Exception:
                    pass
            print("🗑️ All duplicate files deleted.")

    # Show blurry images
    if blur_images:
        print(f"\n😕 Found {len(blur_images)} blurry photos:")
        for img in blur_images:
            print("  -", img)
        delete_blur = input("\nDo you want to delete blurry photos? (y/n): ").lower()
        if delete_blur == "y":
            for img in blur_images:
                try:
                    os.remove(img)
                except Exception:
                    pass
            print("🗑️ Blurry photos deleted.")

    print("\n✨ Cleanup complete!")

if __name__ == "__main__":
    folder = input("📂 Enter folder path to organize & clean: ").strip()
    organize_and_clean(folder)
