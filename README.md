# ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction

**ModelEvalGUI** is a versatile graphical user interface (GUI) application that facilitates the evaluation of machine learning models for three different domains: Email Spam Detection, Cancer Document Classification, and Housing Price Prediction. This project is designed to be user-friendly and flexible, allowing users to choose from various models and evaluation methods for their specific dataset.

## Features

- **Multi-Domain Support**: ModelEvalGUI supports three distinct domains: Email Spam Detection, Cancer Document Classification, and Housing Price Prediction. You can easily choose your dataset of interest.

- **Model Selection**: For each domain, you can select from a range of machine learning models, including K-Nearest Neighbors (KNN), Decision Trees, Naive Bayes (for Email Spam Detection), and K-Means for Cancer Document Classification and Linear Regression for Housing Price Prediction.

- **Evaluation Methods**: Choose between accuracy and confusion matrix as evaluation methods to assess the performance of your selected models.

- **Plotting**: Visualize the evaluation results using the built-in plotting functionality. You can view accuracy or confusion matrices for one or more selected models simultaneously.

- **Model Persistence**: Save your trained machine learning models for later use. ModelEvalGUI provides the option to load these models directly within the application to evaluate and display their accuracy.

## Usage

1. **Select Dataset**: Choose from three datasets - EmailSpam.csv, CancerDocument.csv, and HousingPrice.csv - using the combo box.

2. **Choose Models**: Based on the selected dataset, a list of available models will appear in the form of checkboxes. Select one or more models for evaluation.

3. **Select Evaluation Method**: Pick either accuracy or confusion matrix as the evaluation method.

4. **Plot Results**: Click the "Plot" button to visualize the evaluation results in the figure area.


## Requirements

- Python 3.9
- Required Python libraries (scikit-learn, matplotlib, numpy, pandas, seaborn, emoji, beautifulsoup4, nltk)

## Screenshots

![Interface](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/8fa260f3-af7c-4635-9fb1-a7acb971bcd5)

**Email Spam Detection**



![EmailSpam select all models](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/5c690d2b-e653-4d51-88be-4ddb3ec8775f)
![spamdetectconfusionm](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/cc74272c-7eca-4a20-b663-f98e1d7ffbde)

![Screenshot 2023-09-03 041026](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/152f08e5-509d-4e4a-9ea8-8cb0960f9e41)
![Screenshot 2023-09-03 041051](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/7f5affd8-bca4-4a3f-9353-7742fd326858)

![Screenshot 2023-09-03 041113](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/f64c1314-50d3-4e5e-b399-7a07535f72c0)
![Screenshot 2023-09-03 041135](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/81ab120e-6b00-48a3-8c2c-81c12804e89d)



**Cancer Document Classification**

![Screenshot 2023-09-03 041200](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/eb15b9b0-b7e8-45e2-afed-5c06c03ab2c8)



## Acknowledgments


- [scikit-learn](https://scikit-learn.org/): For providing powerful machine learning tools that i used extensively for model building and evaluation.

- [Matplotlib](https://matplotlib.org/): For creating visualizations, including plots of accuracy and confusion matrices.

- [NumPy](https://numpy.org/): For its fundamental role in scientific computing and data manipulation.

- [pandas](https://pandas.pydata.org/): For simplifying data handling and manipulation tasks, making data analysis more efficient.

- [Seaborn](https://seaborn.pydata.org/): For enhancing the aesthetics of the visualizations and improving the overall look of the graphical representations.

- [Emoji](https://pypi.org/project/emoji/): For expressiveness emojis to a text data.

- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/): For aiding in web scraping tasks and cleaning HTML content.

- [NLTK](https://www.nltk.org/): For its natural language processing capabilities, assisting in text preprocessing and analysis.




**Housing Price Prediction**

![Screenshot 2023-09-03 041224](https://github.com/sara-salah1/ModelEvalGUI-for-EmailSpamDetection-CancerDocumentClassification-and-HousingPricePrediction/assets/67710906/e1d78152-7a77-43d9-b2a6-1dcecda06289)

