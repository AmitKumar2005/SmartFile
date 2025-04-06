from flask import Flask, request, jsonify, send_from_directory
import os
import time
import shutil
import pdfplumber
from docx import Document
from pptx import Presentation
import pickle
from pdf2image import convert_from_path
import pytesseract

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TARGET_BASE_DIR = "C://a"  # Replace with your desired location

# Load the trained model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def extract_text_from_pdf(file_path):
    text = ""
    print(f"Attempting to extract from {file_path}")
    try:
        with pdfplumber.open(file_path) as pdf:
            print(f"Found {len(pdf.pages)} pages with pdfplumber")
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    print(f"Extracted with pdfplumber: {page_text[:50]}...")
                    text += page_text + "\n"
                else:
                    print(
                        f"No text with pdfplumber, trying OCR on page {page.page_number}"
                    )
                    images = convert_from_path(
                        file_path,
                        dpi=400,
                        first_page=page.page_number,
                        last_page=page.page_number,
                    )
                    for img in images:
                        print(f"Processing image for OCR")
                        ocr_text = pytesseract.image_to_string(img, config="--psm 6")
                        print(f"OCR result: {ocr_text[:50]}...")
                        text += ocr_text + "\n"
    except Exception as e:
        print(f"Error in pdfplumber or OCR: {str(e)}")
        try:
            print(f"Falling back to full OCR for {file_path}")
            num_pages = len(pdfplumber.open(file_path).pages)
            for page_num in range(num_pages):
                images = convert_from_path(
                    file_path, dpi=400, first_page=page_num + 1, last_page=page_num + 1
                )
                for img in images:
                    ocr_text = pytesseract.image_to_string(img, config="--psm 6")
                    text += ocr_text + "\n"
        except Exception as e2:
            print(f"OCR fallback failed: {str(e2)}")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    print(f"Final extracted text: {text[:200]}...")
    return text.strip()


def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX {file_path}: {e}")
    return text.strip()


def extract_text_from_pptx(file_path):
    text = []
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
    except Exception as e:
        print(f"Error extracting PPTX {file_path}: {e}")
    return "\n".join(text).strip()


def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".pptx"):
        return extract_text_from_pptx(file_path)
    else:
        return ""


@app.route("/")
def serve_html():
    return send_from_directory(".", "index.html")


@app.route("/extract", methods=["POST"])
def extract():
    print("Hello")
    if "file" not in request.files:
        print("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        print("Empty file")
        return jsonify({"error": "Empty file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print("File saved:", filepath)

    extracted_content = extract_text(filepath)
    if not extracted_content:
        try:
            time.sleep(1)  # Give time for file handles to release
            os.remove(filepath)
        except PermissionError as e:
            print(f"Failed to remove file due to permission (no content): {e}")
            shutil.move(filepath, filepath + ".lock")  # Rename to avoid reuse
        return jsonify({"error": "Could not extract content"}), 400
    print("Extracted Content:", extracted_content)

    X = vectorizer.transform([extracted_content])
    predicted_folder = model.predict(X)[0]
    print("Predicted Folder:", predicted_folder)

    # Move file to custom target location with predicted folder
    target_folder = os.path.join(TARGET_BASE_DIR, predicted_folder)
    target_path = os.path.join(target_folder, file.filename)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created folder: {target_folder}")
    if not os.path.exists(target_path):
        try:
            time.sleep(1)  # Delay to release handles
            shutil.move(filepath, target_path)
            print(f"Moved file to: {target_path}")
        except PermissionError as e:
            print(f"Failed to move file due to permission: {e}")
            shutil.move(filepath, filepath + ".lock")  # Rename if move fails
    else:
        print(f"File {file.filename} already exists in {target_folder}, skipping move")
        try:
            os.remove(filepath)
        except PermissionError as e:
            print(f"Failed to remove file due to permission: {e}")
            shutil.move(filepath, filepath + ".lock")

    response = {"content": extracted_content, "folder": predicted_folder}
    print("Returning JSON:", response)
    return jsonify(response)


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    results = []
    for folder in os.listdir(TARGET_BASE_DIR):
        folder_path = os.path.join(TARGET_BASE_DIR, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith((".pdf", ".docx", ".pptx")):
                    file_path = os.path.join(folder_path, filename)
                    content = extract_text(file_path)
                    if content and query in content.lower():
                        results.append(
                            {
                                "filename": filename,
                                "folder": folder,
                                "content": (
                                    content[:200] + "..."
                                    if len(content) > 200
                                    else content
                                ),
                            }
                        )

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
