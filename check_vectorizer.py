import pickle

file_path = 'tfidf_vectorizer.pkl'  # Change this to each file you're checking

with open(file_path, 'rb') as f:
    obj = pickle.load(f)
    print(f"Loaded object type: {type(obj)}")
    
    # Optional: check if it has a transform method (used by vectorizers)
    if hasattr(obj, 'transform'):
        print("✅ This object can be used for text transformation.")
    else:
        print("❌ This is NOT a vectorizer.")
