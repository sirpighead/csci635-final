# csci635-final
Repo for final project for Machine Learning (CSCI 635)
---

# **KhaanaGPT: Recipe Generation from Ingredients or Images**

## **Introduction**
KhaanaGPT is a state-of-the-art AI project designed to generate recipes based on ingredients or an image of a dish. It leverages advanced machine learning and natural language processing techniques to bridge computer vision and NLP.

Key components:
1. **GPT-2**: Fine-tuned for recipe generation.
2. **CLIP**: Predicts dish names from input images.
3. **Sentence-BERT**: Refines dish predictions through semantic similarity.
4. **Fuzzy Matching**: Handles partial ingredient matches and suggests additional ingredients.

---

## **Features**
### **1. Ingredients to Recipe**
- Input: List of ingredients.
- Output: A generated recipe, including cooking instructions.
- **New Feature**: Fuzzy matching for partial matches and additional ingredient suggestions.

### **2. Image to Recipe**
- Input: Image of a dish.
- Output:
  - Predicted dish name.
  - Ingredients required.
  - Cooking instructions retrieved from the dataset.

### **3. Dataset Integration**
- The **Cleaned Indian Food Dataset** is used, which includes:
  - `TranslatedRecipeName`: Dish names.
  - `Cleaned-Ingredients`: Ingredients list.
  - `TranslatedInstructions`: Step-by-step cooking instructions.

---

## **Dataset**
- **Source**: [Cleaned Indian Food Dataset](https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset).
- **Structure**:
  - Dish Names (`TranslatedRecipeName`).
  - Ingredients (`Cleaned-Ingredients`).
  - Cooking Instructions (`TranslatedInstructions`).

---

## **Requirements**
- **GPU**: Required for efficient training and inference.
- **Python Version**: Python 3.8 or later.
- **Dependencies**:
  - `transformers`
  - `torch`
  - `sentence-transformers`
  - `Pillow`
  - `fuzzywuzzy`
# Note:
# - Training the model on a GPU (like the Google Colab T4 GPU) will be significantly faster than on a CPU.
# - Ensure that your runtime is configured to use a GPU for better performance.
---

## **Installation**
### **Suggested Method**
- Download the .ipynb file and upload it on Google Colab/ Jupyter Notebook.
- Download the dataset from [this link](https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset).
- Save the dataset as `Cleaned_Indian_Food_Dataset.csv` in the root directory.
- Once the model is fully trained upload an image in the same directory where the code is and mention the name of it.
image_path = "Black_Bean_Burrito.jpg"

### **Other Method**
### **Step 1: Clone the Repository**
```bash
https://github.com/sirpighead/csci635-final
```

### **Step 2: Create a Virtual Environment (Optional)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download and Add the Dataset**
- Download the dataset from [this link](https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset).
- Save the dataset as `Cleaned_Indian_Food_Dataset.csv` in the root directory.

### **Step 5: Upload the image**
- Once the model is fully trained upload an image in the same directory where the code is and mention the name of it.
image_path = "Black_Bean_Burrito.jpg"
---

## **How to Run**

### **1. Train the Recipe Generator**
1. Open `ML_RECIPE-GENERATOR.ipynb` in Colab or Jupyter Notebook.
2. Upload the dataset (`Cleaned_Indian_Food_Dataset.csv`).
3. Run all the cells to train the GPT-2 model (`khaanaGPT`) for recipe generation.

### **2. Generate Recipes from Ingredients**
1. After training, input a list of ingredients.
2. Use the `generate_recipe_with_fuzzy_matching` function to generate a recipe.

Example:
```python
user_ingredients = "chicken, tomatoes, garlic"
generate_recipe_with_fuzzy_matching(user_ingredients, df)
```

Output:
```
Based on the ingredients you provided, you can try making: Chicken Curry
Suggested additional ingredients: curry powder, onions
Generated Recipe:
<Recipe Text>
```

### **3. Generate Recipes from Images**
1. Place the image in the project directory.
2. Specify the image path in the `predict_dish_from_image` function.

Example:
```python
image_path = "dish_image.jpg"
predicted_dish = predict_dish_from_image(image_path)
print(f"Predicted Dish: {predicted_dish}")
```

Output:
```
Predicted Dish: Black Bean Burrito
Ingredients: Black beans, tortillas, salsa, onions
Recipe: Heat tortillas, spread black beans and salsa, fold and serve warm.
```

---

## **Code Workflow**

### **Phase 1: Ingredients to Recipe**
1. **Fuzzy Matching**:
   - Matches user-provided ingredients to dataset entries using `fuzzywuzzy`.
   - Suggests additional ingredients if partial matches are found.
2. **Recipe Generation**:
   - Combines user and suggested ingredients.
   - Generates recipe text using the fine-tuned GPT-2 model (`khaanaGPT`).

### **Phase 2: Image to Recipe**
1. **Dish Prediction**:
   - CLIP predicts the dish name from the input image.
   - Sentence-BERT refines predictions through semantic similarity.
2. **Recipe Retrieval**:
   - Fetches ingredients and instructions for the predicted dish from the dataset.
   - Outputs a complete recipe.

---

## **Examples**

### **Example 1: Ingredients to Recipe**
Input:
```python
user_ingredients = "chicken, garlic, tomato"
generate_recipe_with_fuzzy_matching(user_ingredients, df)
```
Output:
```
Based on the ingredients you provided, you can try making: Spicy Chicken Stew
Suggested additional ingredients: onions, paprika, chicken stock
Generated Recipe:
<Recipe Text>
```

### **Example 2: Image to Recipe**
Input:
- Image: `"pasta_dish.jpg"`

Output:
```
Predicted Dish: Alfredo Pasta
Ingredients: Pasta, cream, garlic, parmesan cheese
Recipe Instructions:
Boil pasta until al dente. Heat cream and garlic in a pan, stir in parmesan cheese. Toss pasta in the sauce and serve warm.
```

---

## **Acknowledgments**
- **Hugging Face** for GPT-2, CLIP, and `transformers` library.
- **Kaggle** for providing the Indian Food Dataset.
- **Colab** for enabling GPU-powered training and inference.

---
