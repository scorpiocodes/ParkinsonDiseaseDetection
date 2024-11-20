# ParkinsonDiseaseDetection

## What is Parkinson's Disease?

Parkinson's disease (PD), or simply Parkinson's, is a long-term degenerative disorder of the central nervous system that mainly affects the motor system. As the disease worsens, non-motor symptoms become more common. The symptoms usually emerge slowly. Early in the disease, the most obvious symptoms are shaking, rigidity, slowness of movement, and difficulty with walking. Thinking and behavioral problems may also occur. Dementia becomes common in the advanced stages of the disease. Depression and anxiety are also common, occurring in more than a third of people with PD. Other symptoms include sensory, sleep, and emotional problems. The main motor symptoms are collectively called "parkinsonism", or a "parkinsonian syndrome".

![Dieseased Person](img/image1.png)

## Cause of the disease!

The cause of Parkinson's disease is unknown, but is believed to involve both genetic and environmental factors. Those with a family member affected are more likely to get the disease themselves.

* Genetics
* Environmental factors

Exposure to pesticides and a history of head injury have each been linked with Parkinson disease (PD), but the risks are modest. Never having smoked cigarettes, and never drinking caffeinated beverages, are also associated with small increases in risk of developing PD.
Low concentrations of urate in the blood serum is associated with an increased risk of PD.

![parkinson's](img/PD.png)

## Applied Machine Learning in Healthcare.

Machine learning in medicine has recently made headlines. Google has developed a machine learning algorithm to help identify cancerous tumors on mammograms. Stanford is using a deep learning algorithm to identify skin cancer. A recent JAMA article reported the results of a deep machine-learning algorithm that was able to diagnose diabetic retinopathy in retinal images. It’s clear that machine learning puts another arrow in the quiver of clinical decision making.

Still, machine learning lends itself to some processes better than others. Algorithms can provide immediate benefit to disciplines with processes that are reproducible or standardized. Also, those with large image datasets, such as radiology, cardiology, and pathology, are strong candidates. Machine learning can be trained to look at images, identify abnormalities, and point to areas that need attention, thus improving the accuracy of all these processes. Long term, machine learning will benefit the family practitioner or internist at the bedside. Machine learning can offer an objective opinion to improve efficiency, reliability, and accuracy.

![MachineLearning_PersonalizedMedicine](img/MachineLearning_PersonalizedMedicine.jpg)

## Data Drives Machine Learning.

As more data is available, we have better information to provide patients. Predictive algorithms and machine learning can give us a better predictive model of mortality that doctors can use to educate patients.

But machine learning needs a certain amount of data to generate an effective algorithm. Much of machine learning will initially come from organizations with big datasets. Health Catalyst is developing Collective Analytics for Excellence (CAFÉ™), an application built on a national de-identified repository of healthcare data from enterprise data warehouses (EDWs) and third-party data sources. It is enabling comparative effectiveness, research, and producing unique, powerful machine learning algorithms. CAFÉ provides a collaboration among our healthcare system partners, big and small.

As larger datasets begin to run machine learning, we can improve care in more specific ways for each region. And considering rare diseases with low data volumes, it should be possible to merge regional data into national sets to scale the volume needed for machine learning.

## What is XGBoost?

XGBoost is a new Machine Learning algorithm designed with speed and performance in mind. XGBoost stands for eXtreme Gradient Boosting and is based on decision trees. In this project, we will import the XGBClassifier from the xgboost library; this is an implementation of the scikit-learn API for XGBoost classification.

![XGB](img/image2.png)


## Project Overview

Parkinson's disease is a progressive neurological disorder that affects movement control. Early detection is crucial for effective management and treatment. This project utilizes machine learning techniques to analyze voice recordings and identify patterns indicative of Parkinson's disease.

## Features

- **Data Preprocessing**: Cleans and prepares the dataset for analysis.
- **Model Training**: Implements various machine learning algorithms to train the detection model.
- **Model Evaluation**: Assesses the performance of the trained models using appropriate metrics.
- **Web Application**: Provides a user-friendly interface for inputting numeric voice data and obtaining predictions.

## Dataset

The project utilizes the [Parkinson's Disease dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/), which includes biomedical voice measurements from 31 individuals, 23 with Parkinson's disease. The dataset comprises 197 instances and 23 attributes, such as vocal fundamental frequency, variation in amplitude, and noise-to-tonal component ratios.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/scorpiocodes/ParkinsonDiseaseDetection.git
   cd ParkinsonDiseaseDetection
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:

   ```bash
   python app.py
   ```

   The application will start, and you can access it by navigating to `http://localhost:5000` in your web browser.

## Usage

Once the application is running, you can input voice measurement data through the web interface to receive a prediction on the likelihood of Parkinson's disease.

---

*Note: Ensure you have Python 3.x installed on your machine before setting up the project.* 
&copy;HR
