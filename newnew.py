from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model("D:/dl project_new_new/Traffic_Sign_Detection/Traffic_Sign_Detection/Traffic_Signs_Detection_vgg_2.h5")

# Define the classes of traffic signs
classes = {
    0: 'Ahead only',
    1: 'Beware of icesnow',
    2: 'Bicycles crossing',
    3: 'Bumpy road',
    4: 'Children crossing',
    5: 'Dangerous curve left',
    6: 'Dangerous curve right',
    7: 'Double curve',
    8: 'End of no passing',
    9: 'End no passing vehicle 3.5 tons'
}

while True:
    # Take input image path from the user
    image_path = input("Enter the path of the input image (or type 'exit' to quit): ")
    
    # Check if the user wants to exit
    if image_path.lower() == 'exit':
        print("Exiting...")
        break
    
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        image = image.resize((37, 37))  # Resize the image to match the input size of the model
        data = np.array(image)
        X_test = np.expand_dims(data, axis=0)  # Add batch dimension

        # Make predictions
        Y_pred = np.argmax(model.predict(X_test), axis=1)
        predicted_class = classes[Y_pred[0]]

        # Display the input image along with the predicted output class
        plt.imshow(image)
        plt.title(f"Predicted traffic sign: {predicted_class}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print("Error:", e)
