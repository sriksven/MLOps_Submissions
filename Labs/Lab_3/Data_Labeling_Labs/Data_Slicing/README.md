# Snorkel Data Slicing with IMDb Movie Reviews

## 1. Objective
This notebook adapts the Snorkel Data Slicing tutorial to a new dataset, IMDb Movie Reviews, to identify and improve model performance on specific data slices (subsets of examples).  
The main goal is to analyze how the model performs on certain kinds of text such as:

- Short reviews (less than 30 words)
- Very positive reviews (sentiment polarity > 0.8)

This experiment demonstrates that even when overall model accuracy is high, performance can vary significantly across data subsets.

---

## 2. Dataset
- Dataset used: IMDb Movie Reviews from Hugging Face (`datasets.load_dataset("imdb")`).
- Contains 50,000 movie reviews labeled as positive (1) or negative (0).
- Train and test splits: 25,000 examples each.
- Reviews vary widely in tone, sentiment, and length, making them suitable for defining meaningful slices.

---

## 3. Slicing Functions
Two programmatic slicing functions were defined to automatically mark certain reviews for closer monitoring.

```python
@slicing_function()
def short_review(x):
    return len(x.text.split()) < 30

@slicing_function()
def very_positive(x):
    polarity = TextBlob(x.text).sentiment.polarity
    return polarity > 0.8
```

Each function outputs a binary mask (1 if the review belongs to the slice, 0 otherwise).  
These slices identify subsets of the dataset that may have distinct patterns or challenges for the classifier.

---

## 4. Base Model
- Features extracted using `CountVectorizer` with unigrams and bigrams.
- A Logistic Regression model was trained for sentiment classification.
- Baseline performance:
  - Overall F1 score: approximately 0.90–0.93
  - Short review F1 score: approximately 0.70
  - Very positive review F1 score: approximately 0.80

The baseline showed that while the overall performance was good, some slices performed significantly worse.

---

## 5. Applying Slicing Functions
The slicing functions were applied using Snorkel’s `PandasSFApplier`, producing binary matrices `S_train` and `S_test` that record which samples belong to each slice.  
These matrices are used by Snorkel’s slice-aware training methods to link samples with their corresponding slice information.

---

## 6. Slice-Aware Learning
To improve per-slice performance, the model was extended with Snorkel’s `SliceAwareClassifier`, which performs slice-based learning.  
Key steps included:

- Implementing a simple feed-forward encoder (`BoWEncoder`) that outputs 256-dimensional embeddings from the bag-of-words features.
- Using `SliceAwareClassifier` to add slice-specific heads for the two defined slices.
- Training the model with Snorkel’s `Trainer` for two epochs using a batch size of 64.

This approach combines multi-task learning principles, allowing the model to learn shared representations while focusing additional capacity on slices that may require special handling.

---

## 7. Results and Observations
After training with the `SliceAwareClassifier`:
- Overall F1 score remained stable around 0.93.
- The `short_review` slice F1 score improved by approximately 5–10 percent.
- The `very_positive` slice F1 score remained strong around 0.82 or higher.

The results confirm that slice-aware learning improves model robustness on underperforming slices without sacrificing overall accuracy.

---

## 8. Key Learnings
- Slicing functions allow targeted monitoring of model performance on important subsets.
- The `SliceAwareClassifier` integrates slice information into the learning process to enhance performance on specific groups of data.
- High overall accuracy can mask weaknesses on smaller but important subsets.
- This workflow demonstrates how data slicing can help build fairer and more reliable models.

---

## 9. Summary
This notebook applied the Snorkel Data Slicing framework to the IMDb Movie Reviews dataset.  
Two slicing functions were implemented to monitor short and very positive reviews.  
A logistic regression model was trained to establish a baseline, and then a slice-aware model was developed using Snorkel’s `SliceAwareClassifier`.  
Results showed that targeted slice learning improves the model’s robustness and interpretability by identifying and addressing subset-specific weaknesses.
