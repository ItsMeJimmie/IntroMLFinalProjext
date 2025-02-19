# IntroMLFinalProjext
Below you can see our explanation of our Terrain Image Segmentation work we did.

# General Description
This is the public repository of our Terrain Image Segmentation project that we did for our final project of our Intro To Machine Learning class. The class is CS4840. The people who worked on this project are Jimmie Cox (Repo Owner) and Vichaka Houi. The project was completed in a month which included collecting research, writing code, creating the report, and making the presentation. The entire project lasted about the entirety of November 2024 to the beginning of December 2024. 

Instead of manually typing out what we did for the project, here's the abstract of our paper we wrote. 

This study addresses the challenge of classifying terrain images and its type from satellite imagery whilst using machine learning techniques. Accurate terrain classification is crucial for applications such as monitoring city development and urban development or monitoring changes in an environment. The research used “Earth Terrain, Height, and Segmentation Map Images” by Thomas Pappas on Kaggle comprising of 15,000 images divided into to each of the following: terrain, height and masking image (5000 each). Using feature extraction, RGB values were extracted from the terrain and height features from the height image. Due to computational restraints, the dataset was decreased to 3,000 and images were reduced from 512 x 512 to 64 x 64 pixels. Three different training models were used: Support Vector Machine (SVM), Random Forest, K-nearest neighbors (KNN). The SVM model had achieved an overall accuracy of 94%, with micro precision being at 91%, micro f1 at 93% and micro recall at 93%. It also had macro evaluations at 91% for precision, 94% for recall, and 92% for f1. While SVM performz well, when compared to models like KNN and random forest. Both models can work and achieve near perfect scores in all evaluation metrics. Although KNN is slower, but better for smaller datasets. With random forest, tuning is necessary to achieve such results.

Keywords: Support Vector Machine, Random Forest, K-Nearest Neighbors

The actual code, report, and presentation can be found in this repository as well. 

Final Grade: 99%

