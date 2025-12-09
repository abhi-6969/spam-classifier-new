import joblib
import tkinter as tk
from tkinter import ttk, Text, scrolledtext

# Load model
model = joblib.load('spam_classifier_model.pkl')

def check_spam():
    text = text_box.get("1.0", tk.END).strip()
    prediction = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    
    result = "SPAM" if prediction == 1 else "NOT SPAM"
    confidence = max(prob) * 100
    
    result_label.config(text=f"Result: {result}\nConfidence: {confidence:.1f}%",
                        foreground="red" if prediction == 1 else "green")

# GUI
root = tk.Tk()
root.title("Email/SMS Spam Classifier")
root.geometry("600x500")
root.configure(padx=20, pady=20)

tk.Label(root, text="Enter Email or SMS Text:", font=("Arial", 14)).pack(pady=10)

text_box = scrolledtext.ScrolledText(root, height=15, font=("Arial", 12))
text_box.pack(padx=10, fill=tk.BOTH, expand=True)

btn = ttk.Button(root, text="Check if Spam", command=check_spam)
btn.pack(pady=15)

result_label = tk.Label(root, text="Result will appear here", font=("Arial", 16), height=3)
result_label.pack()

root.mainloop()