# PlantView

This repository provides the core code needed for the PlantView project, designed to support data processing and augmentation for research and model training.

Dataset Access
The initial dataset required for this project can be downloaded from the following link: https://drive.google.com/drive/folders/1o0sxsEEbRgiZpBpY7hALFkCQfxUcI35b?usp=drive_link. This dataset serves as a starting point for research and data generation purposes.

Code Overview: PlantView.py
The PlantView.py script is a powerful tool for processing and augmenting the initial dataset, allowing researchers to expand the volume and diversity of data samples for enhanced analysis and model robustness.

Key Augmentation Steps in PlantView.py
The following augmentation steps are included or supported in the PlantView.py script to increase the dataset size and variety:

Rotation: Random rotations applied to each image or data sample to simulate different orientations.
Flipping: Horizontal and vertical flipping to enhance spatial diversity.
Scaling: Adjusting the scale of samples to simulate various distances or perspectives.
Noise Injection: Adding random noise to simulate real-world data variability.
Color Adjustment: Modifying brightness, contrast, and saturation for enhanced variation.
These augmentation techniques collectively improve the dataset's representational power, making it suitable for training more robust models and enhancing research outcomes.

Usage Instructions
Download the Data: Download the initial dataset from the provided link.
Run PlantView.py: Use the script to process the data and apply augmentation techniques.
Review the Augmented Dataset: The augmented data will contain a higher number of diverse samples, ready for further analysis or model training.
