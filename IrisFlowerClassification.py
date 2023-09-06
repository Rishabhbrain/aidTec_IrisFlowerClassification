import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
# Load the Iris dataset (Make sure to upload 'iris.csv' to your Colab environment)
data = pd.read_csv('/content/IRIS.csv')

# Split the data into features (X) and target (y)
X = data.drop('species', axis=1)
y = data['species']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# Function to predict the Iris species based on user input
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create a prediction input
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Get the predicted species
    predicted_species = clf.predict(input_data)

    return predicted_species[0]
def predict_species_ui(button):
    try:
        # Get user input
        sl = float(input_sepal_length.value)
        sw = float(input_sepal_width.value)
        pl = float(input_petal_length.value)
        pw = float(input_petal_width.value)

        # Get the predicted species
        species = predict_species(sl, sw, pl, pw)

        # Clear previous output
        clear_output()

        # Display the result
        display(HTML("<h2>Predicted Iris Species</h2>"))
        display(HTML(f"<p>Species: {species}</p>"))

    except ValueError:
        print("Please enter valid numeric values.")
# Create UI elements
input_sepal_length = widgets.FloatText(description="Sepal Length")
input_sepal_width = widgets.FloatText(description="Sepal Width")
input_petal_length = widgets.FloatText(description="Petal Length")
input_petal_width = widgets.FloatText(description="Petal Width")
predict_button = widgets.Button(description="Predict")
predict_button.on_click(predict_species_ui)

# Display UI
display(input_sepal_length, input_sepal_width, input_petal_length, input_petal_width, predict_button)
