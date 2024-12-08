# csci635-final
Repo for final project for Machine Learning (CSCI 635)

# **KhaanaGPT: Recipe Generation from Ingredients or Images**

## **Introduction**
KhaanaGPT is a deep learning-powered project designed to generate cooking recipes based on ingredients or an image of a dish. It combines the power of:
1. **GPT-2** (fine-tuned to generate recipes based on input ingredients).
2. **CLIP** (to predict dish names from input images).
3. **Sentence-BERT** (for semantic similarity matching to refine dish predictions).

This project bridges computer vision and natural language processing to produce a seamless flow from image to recipe.

---

## **Features**
1. **Image to Recipe**:
   - Input: Image of a dish.
   - Output: Predicted dish name, ingredients, and recipe.

2. **Ingredients to Recipe**:
   - Input: List of ingredients.
   - Output: Generated recipe.

3. **Integrated Dataset**:
   - Dataset containing dishes, ingredients, and instructions for fine-tuning GPT-2.

4. **Modular Design**:
   - Fine-tuned GPT-2 for recipe generation.
   - CLIP for image-text matching.
   - Sentence-BERT for semantic similarity.

---

## **Dataset**
The project uses the **Cleaned Indian Food Dataset**, which contains:
- Dish names (`TranslatedRecipeName`),
- Ingredients (`Cleaned-Ingredients`),
- Instructions (`TranslatedInstructions`).

**Dataset Link**: [Cleaned Indian Food Dataset](https://www.kaggle.com/datasets/saldenisov/recipenlg/data)

---

## **Requirements**
- **GPU**: Required for training and efficient inference of models.
- **Python Version**: Python 3.8 or later.
- **Packages**:
  - `transformers`
  - `datasets`
  - `torch`
  - `sentence-transformers`
  - `Pillow`

---

## **Installation**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your_username/khaanaGPT.git
cd khaanaGPT
```

### **Step 2: Create a Virtual Environment (Optional)**
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\\Scripts\\activate     # On Windows
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download the Dataset**
- Download the dataset from [this link](https://www.kaggle.com/datasets/saldenisov/recipenlg/data).
- Save it in the root directory as `Cleaned_Indian_Food_Dataset.csv`.

---

## **How to Run**

### **1. Train the Recipe Generator (KhaanaGPT)**
1. Open the Jupyter notebook or Colab file (`ML_PROJECT_PHASE_1&2_WORKING.ipynb`).
2. Upload the dataset (`Cleaned_Indian_Food_Dataset.csv`).
3. Run the notebook to train `khaanaGPT` for recipe generation.

### **2. Generate Recipes from Ingredients**
Once the model is trained:
1. Input a list of ingredients.
2. Use the `generate_recipe` function to generate a recipe.

Example:
```python
ingredients = "chicken, tomatoes, curry powder, garlic, onions"
generate_recipe(ingredients)
```

### **3. Generate Recipes from Images**
1. Save the image in the project directory.
2. Specify the image path in the `predict_dish_from_image` function.

Example:
```python
image_path = "dish_image.jpg"
predicted_dish = predict_dish_from_image(image_path)
print(f"Predicted Dish: {predicted_dish}")
```

---

## **Code Workflow**
1. **Dataset Preparation**:
   - Preprocess dataset into input-output pairs for GPT-2.

2. **Training**:
   - Fine-tune GPT-2 for recipe generation.

3. **Image Processing**:
   - Use CLIP to predict the dish name from the input image.
   - Refine the prediction using Sentence-BERT.

4. **Recipe Generation**:
   - Retrieve ingredients from the dataset for the predicted dish.
   - Use `khaanaGPT` to generate the recipe.

---

## **Examples**

### **Example 1: Generate Recipe from Ingredients**
Input:
```python
ingredients = "potatoes, tomatoes, onions, garlic"
generate_recipe(ingredients)
```
Output:
```
Heat oil in a pan. Add onions and garlic, sauté until golden. Add tomatoes and potatoes. Cook until tender. Serve hot!
```

### **Example 2: Generate Recipe from Image**
Input:
- Image: `"Black_Bean_Burrito.jpg"`

Output:
```
Predicted Dish: Black Bean Burrito
Ingredients: Black beans, tortillas, salsa, onions
Recipe: Heat tortillas on a pan. Spread black beans, salsa, and onions on the tortilla. Fold and serve warm.
```

---

## **Future Improvements**
1. Extend dataset to include global cuisines.
2. Improve accuracy of dish prediction using larger models.
3. Integrate real-time image capture and recipe generation.

---

## **Contributions**
- Contributions are welcome! Feel free to submit a pull request or open an issue for feedback.

---

## **License**
This project is licensed under the MIT License. 

---

Copy this content and save it to a file named `README.md`.
