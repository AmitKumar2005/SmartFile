import os
import shutil
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import AdamW


# 1. File Content Extraction (Example for PDFs)
def extract_pdf_text(file_path):
    import PyPDF2

    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return text


# 2. Data Preparation
def prepare_data(data_dir):
    data = []
    labels = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_path.lower().endswith((".pdf", ".docx", ".txt")):
                    if file_path.lower().endswith(".pdf"):
                        text = extract_pdf_text(file_path)
                    else:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                text = f.read()
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
                            text = ""
                    data.append(text)
                    labels.append(folder_name)
    return data, labels


# 3. TF-IDF and SVM (Baseline)
def train_tfidf_svm(data, labels):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"TF-IDF SVM Accuracy: {accuracy}")
    return vectorizer, model


# 4. DistilBERT (Advanced)
# def train_distilbert(data, labels):
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(labels)))
#     encoded_data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
#     label_ids = torch.tensor([list(set(labels)).index(label) for label in labels])
#     dataset = TensorDataset(encoded_data['input_ids'], encoded_data['attention_mask'], label_ids)
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=8)
#     optimizer = AdamW(model.parameters(), lr=1e-5)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model.to(device)
#     model.train()
#     for epoch in range(3): # Adjust epochs as needed
#         for batch in train_loader:
#             input_ids, attention_mask, label_batch = tuple(t.to(device) for t in batch)
#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask=attention_mask, labels=label_batch)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids, attention_mask, label_batch = tuple(t.to(device) for t in batch)
#             outputs = model(input_ids, attention_mask=attention_mask)
#             _, predicted = torch.max(outputs.logits, 1)
#             total += label_batch.size(0)
#             correct += (predicted == label_batch).sum().item()
#     print(f"DistilBERT Accuracy: {correct / total}")
#     return tokenizer, model


# 5. Rename and Move File
def rename_and_move_file(file_path, destination_path):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.basename(destination_path)
    new_file_name = f"{folder_name}_{timestamp}{os.path.splitext(file_path)[1]}"
    new_file_path = os.path.join(destination_path, new_file_name)
    shutil.move(file_path, new_file_path)
    return new_file_path


# Example Usage
data_dir = "C:/Users/admin/OneDrive/Documents"  # replace with your data directory.
data, labels = prepare_data(data_dir)

# Choose your model
vectorizer, svm_model = train_tfidf_svm(data, labels)
# tokenizer, distilbert_model = train_distilbert(data, labels)

# Example file prediction and renaming.
test_file_path = (
    "C:/Users/admin/OneDrive/Documents/WT_8to12.pdf"  # replace with test file path
)
text = extract_pdf_text(test_file_path)

# Example prediction. Adapt according to what model you choose.
if "svm_model" in locals():
    test_vector = vectorizer.transform([text])
    predicted_folder = svm_model.predict(test_vector)[0]
# else:
#     tokenizer_test = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     distilbert_model.to(device)
#     distilbert_model.eval()
#     with torch.no_grad():
#         outputs = distilbert_model(tokenizer_test['input_ids'].to(device), attention_mask=tokenizer_test['attention_mask'].to(device))
#         _, predicted = torch.max(outputs.logits, 1)
#         predicted_folder = list(set(labels))[predicted.item()]

destination_path = os.path.join(data_dir, predicted_folder)
os.makedirs(destination_path, exist_ok=True)  # create folder if does not exist.
new_file_path = rename_and_move_file(test_file_path, destination_path)
print(f"File moved to: {new_file_path}")
