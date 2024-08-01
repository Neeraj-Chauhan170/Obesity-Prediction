import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


st.set_page_config(layout="wide")
    
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Define all functions upfront
def load_and_preprocess_data():
    file_path = "ObesityDataSet_raw_and_data_sinthetic.csv"
    data = pd.read_csv(file_path)
    # Encode binary and categorical columns, calculate BMI
    binary_columns = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    categorical_columns = ['Gender', 'CAEC', 'CALC', 'MTRANS']
    for column in binary_columns:
        data[column] = data[column].map({'no': 0, 'yes': 1})
    
    le = LabelEncoder()
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column])

    data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    return data

def split_data(data):
    X = data.drop(columns='NObeyesdad')  
    y = data['NObeyesdad']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm, title):
    col1, col2, col3, col4, col5= st.columns([1,1, 8, 1, 1])
    with col3:
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion matrix')
        st.pyplot(fig)
        #plt.close()

def plot_correlation_heatmap(data):
    # Filter the data to include only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()
    fig = px.imshow(correlation, text_auto=True, aspect="auto", color_continuous_scale='Viridis', title='Correlation Heatmap')
    fig.update_layout(width=900, height=800)
    return fig

# Define the shap_values_to_list function
def shap_values_to_list(shap_values, model):
    shap_as_list = []
    for i in range(len(model.classes_)):
        shap_as_list.append(shap_values[:, :, i])
    return shap_as_list


def plot_shap_summary(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_as_list = shap_values_to_list(shap_values, model)
    
    st.write("#### **SHAP Summary Plot**")
    st.caption('###### The summary plot shows the feature importance of each feature in the model.')
    col1, col2, col3, col4, col5= st.columns([1,1, 6, 1, 1])
    with col3:
        shap.summary_plot(shap_as_list, X_test, plot_type="bar", class_names=model.classes_)
        plt.tight_layout()
        st.pyplot(bbox_inches='tight')
    st.info(" The summary plot shows that **BMI, Weight, Gender, FCVC** play major contributions in determining the obesity level.", icon="ℹ️")

def plot_shap_summary_class(model, X_test, class_index, class_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    st.write(f"#### **SHAP Summary Plot for Class: {class_names[class_index]}**")
    st.caption('###### Using the sidebar in the navigation bar, you can visualise the summary plot for each class of obesity level.')
    col1, col2, col3, col4, col5= st.columns([1,1, 6, 1, 1])

    with col3:
        shap.summary_plot(shap_values[:, :, class_index], X_test, plot_type='bar', class_names=[class_names[class_index]])
        plt.tight_layout()
        st.pyplot(bbox_inches='tight')

def shap_decision_plot(model, X_test, y_test, index, class_names):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_as_list = shap_values_to_list(shap_values, model)

    class_index = class_names.index(y_test.iloc[index])
    
    col1, col2, col3, col4, col5= st.columns([1,1, 6, 1, 1])
    with col3:
        # Plot the SHAP decision plot for the specified class and instance index
        shap.decision_plot(explainer.expected_value[class_index], shap_as_list[class_index][index], X_test.iloc[index])
        plt.tight_layout()
        st.pyplot(bbox_inches='tight')
    
   
    
def get_variable_descriptions():
    descriptions = {
        "Variable Name": ["Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS", "NObeyesdad"],
        "Description": [
            "Gender of the individual", "Age of the individual", "Height in meters", "Weight in kg", "Family history of overweight",
            "Frequent consumption of high caloric food", "Frequency of vegetables in meals", "Number of main meals",
            "Consumption of food between meals", "Smoking status", "Daily water consumption in liters",
            "Calorie intake monitoring", "Physical activity frequency", "Time using technological devices",
            "Alcohol consumption frequency", "Mode of transportation used", "Classification of obesity level"
        ],
        "Type": ["Categorical", "Continuous", "Continuous", "Continuous", "Binary", "Binary", "Integer", "Continuous", "Categorical", "Binary", "Continuous", "Binary", "Continuous", "Integer", "Categorical", "Categorical", "Categorical"]
    }
    return pd.DataFrame(descriptions)

# Load and preprocess data
data = load_and_preprocess_data()
X_train, X_test, y_train, y_test = split_data(data)
# Standardize the features

#loading the model
with open('RF_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)


#-------------------- Interface code --------------------------------------

# Streamlit Sidebar for Navigation
with st.sidebar:

    selection = option_menu("Navigation Menu", ["Home",  "Dataset Overview & Exploration", "Model Performance", "Explainability", "Obesity Estimation Tool", "Personalized Recommendations", "User Feedback"],
                            icons=['house', 'table', 'bar-chart-line', 'lightbulb', 'calculator', 'person-lines-fill' ,'envelope'],
                            menu_icon="grid-fill", default_index=0, styles={
        "container": {"padding": "2!important"}, 
        "icon": {"font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fd7e14"},  
        "nav-link-selected": {"background-color": "#fd7e14"}, 
    })


if selection == "Home":
    st.header("Obesity Level Estimation Dashboard")
    st.divider()
    
    # Purpose of the dataset
    st.subheader("Purpose of the Dataset")
    st.write("The primary purpose of this dataset is to estimate obesity levels based on eating habits and physical condition.")
    st.markdown("[Link to the UCI Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)")

    # Why we chose this dataset
    st.subheader("Why We Chose This Dataset")
    st.write("""
    Obesity is a significant public health concern, and understanding the factors that contribute to it can help in developing effective interventions. 
    This dataset provides comprehensive information on various lifestyle and health-related factors that can be used to predict obesity levels.
    """)


    # Additional Information
    st.subheader("Information on Obesity Risks")
    st.write("""
    Obesity is associated with a higher risk of numerous health issues, including:
    - Cardiovascular diseases
    - Type 2 diabetes
    - Certain types of cancer
    - Hypertension
    - Stroke
    """)

    
    # Risk Assessment Tool Information
    st.subheader("Risk Assessment Tool")

    
    st.write("""
    This dashboard includes a risk assessment tool designed to provide insights into an individual's obesity risk based on their lifestyle and health data. 
    The tool uses machine learning models to predict obesity levels and offers personalized recommendations to help manage and reduce obesity risk.
    """)
    
    # # Image or Diagram (Optional)
    # col1, col2, col3, col4, col5= st.columns([1,1, 10, 1, 1])
    # with col3:
    #     st.image("DIGITAL.webp", caption="Raising awareness on potential risks of Obesity", use_column_width=True)

    # Any additional info or links
    st.write("""
    For more information, visit [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight) or 
    [Centers for Disease Control and Prevention](https://www.cdc.gov/obesity/index.html).
    """)
    
elif selection == "Dataset Overview & Exploration":
    st.header("Dataset Overview & Exploration")
    st.divider()
    
    # Concise introduction about data collection
    st.write("""
    ### Dataset Collection Overview
    **Collection Methods and Sources:**
    - **Countries Involved**: Mexico, Peru, and Colombia
    - **Synthetic Data**: 77% of the data was generated using the SMOTE filter in Weka tool.
    - **Direct Collection**: 23% of the data was collected from users through a web platform.
    """)

    # Load the original dataset
    original_data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

    st.subheader("Variable Descriptions")
    st.info('NObeyesdad is the Target Variable rest all are Feature Variables', icon="ℹ️")
    descriptions = get_variable_descriptions()
    st.table(descriptions)
    
    st.subheader("Filter Dataset")
    col1, col2 = st.columns([1, 2])

    with col1:
        gender_filter = st.multiselect("Select Gender:", options=original_data["Gender"].unique(), default=original_data["Gender"].unique())
        fam_history_filter = st.multiselect("Family History of Overweight:", options=original_data["family_history_with_overweight"].unique(), default=original_data["family_history_with_overweight"].unique())
        smoke_filter = st.multiselect("Smoking Status:", options=original_data["SMOKE"].unique(), default=original_data["SMOKE"].unique())
        mtrans_filter = st.multiselect("Mode of Transportation:", options=original_data["MTRANS"].unique(), default=original_data["MTRANS"].unique())
        obesity_filter = st.multiselect("Obesity Level:", options=original_data["NObeyesdad"].unique(), default=original_data["NObeyesdad"].unique())
        
        # Double-ended sliders for continuous variables
        age_min, age_max = st.slider("Select Age Range:", min_value=int(original_data["Age"].min()), max_value=int(original_data["Age"].max()), value=(int(original_data["Age"].min()), int(original_data["Age"].max())))
        height_min, height_max = st.slider("Select Height Range (m):", min_value=float(original_data["Height"].min()), max_value=float(original_data["Height"].max()), value=(float(original_data["Height"].min()), float(original_data["Height"].max())))
        weight_min, weight_max = st.slider("Select Weight Range (kg):", min_value=float(original_data["Weight"].min()), max_value=float(original_data["Weight"].max()), value=(float(original_data["Weight"].min()), float(original_data["Weight"].max())))

    with col2:
        # Apply filters
        filtered_data = original_data[
            (original_data["Gender"].isin(gender_filter)) &
            (original_data["family_history_with_overweight"].isin(fam_history_filter)) &
            (original_data["SMOKE"].isin(smoke_filter)) &
            (original_data["MTRANS"].isin(mtrans_filter)) &
            (original_data["NObeyesdad"].isin(obesity_filter)) &
            (original_data["Age"] >= age_min) & (original_data["Age"] <= age_max) &
            (original_data["Height"] >= height_min) & (original_data["Height"] <= height_max) &
            (original_data["Weight"] >= weight_min) & (original_data["Weight"] <= weight_max)
        ]
        
        st.write("Here's the filtered overview of the dataset:")
        st.dataframe(filtered_data.head(10))

        st.write("Here's the summary of the filtered dataset:")
        st.write(filtered_data.describe())

    col1, col2, col3 = st.columns(3)
        
    with col1:
        fig = px.pie(filtered_data, names='family_history_with_overweight', title='Family History of Overweight', color_discrete_sequence=['#FF4B60', '#FFDFE5'])
        st.plotly_chart(fig)
        
    with col2:
        fig = px.pie(filtered_data, names='SMOKE', title='Smoking Status', color_discrete_sequence=['#33C0FF', '#E3F7FF'])
        st.plotly_chart(fig)

    with col3:
        fig = px.pie(filtered_data, names='Gender', title='Gender Distribution', color_discrete_sequence=['#A5C916', '#F3F7D9'])
        st.plotly_chart(fig)


    st.header("Explore the Data")

    col2 = st.columns(1)[0]
    with col2:
        fig = px.histogram(filtered_data, x='NObeyesdad', title='Obesity Level Distribution', color='NObeyesdad', text_auto=True)
        st.plotly_chart(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap of Variables")
    st.plotly_chart(plot_correlation_heatmap(data))

    # BMI Distribution by Obesity Levels
    fig = px.box(data, x='NObeyesdad', y='BMI', title='BMI Distribution by Obesity Levels', color='NObeyesdad')
    st.plotly_chart(fig)

    # Physical Activity Frequency by Obesity Levels
    fig = px.box(data, x='NObeyesdad', y='FAF', title='Physical Activity Frequency by Obesity Levels', color='NObeyesdad')
    st.plotly_chart(fig)


elif selection == "Model Performance":
    
    st.header("Model Performance Insights")
    st.divider()
    # Random Forest Model Insights
    st.subheader("Confusion Matrix")
    st.caption('The confusion matrix helps to understand the accuracy of a classification model by showing how many instances of each class were correctly or incorrectly classified.')
    cm_dt = confusion_matrix(y_test, rf_model.predict(X_test))
    plot_confusion_matrix(cm_dt, "Random Forest Classifier Confusion Matrix")
    #st.pyplot(fig)

    # Model performance metrics
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    #accuracy score
    st.info(f"Model Accuracy on Test Data: {accuracy:.2f}", icon="🎯")

    # Classification Report
    st.subheader("Classification Report")
    st.caption('A classification report is a comprehensive way to evaluate the performance of a classification model. It provides key metrics for each class, including precision, recall, F1-score, and support. These metrics help in understanding how well your classification model is performing across different classes.')
    report_df = pd.DataFrame(report).transpose()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(report_df)
    with col2:
        st.write("**Key Metrics:**")
        st.latex(r'''
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
''')
        st.latex(r'''
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
''')
        st.latex(r'''
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
''')






elif selection == "Explainability":

    st.header("Model Explanations")
    st.divider()

    st.write("""
    ### SHAP values in Model Interpretability
    - SHAP values are a common way of getting a consistent and objective explanation of how each feature impacts the model's prediction.
    - Features with positive SHAP values positively impact the prediction, while those with negative values have a negative impact. 
    - The magnitude is a measure of how strong the effect is.
    """)
    

    # Define class names
    class_names = ['Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
               'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
               'Obesity_Type_III']

    # SHAP values for Decision Tree
    plot_shap_summary(rf_model, X_test)
    
    # Dropdown for class selection
    selected_class = st.sidebar.selectbox("## **Select a class to explain**", class_names)
    selected_class_index = class_names.index(selected_class)
    
    plot_shap_summary_class(rf_model, X_test, selected_class_index, class_names)
    st.write("The features with larger SHAP values in the summary plot of each class contribute more towards the prediction of the particular class")


    st.write("#### **SHAP Decision Plot**")
    st.caption("The SHAP decision plot visualizes how features contribute to a particular prediction. It helps in understanding the impact of each feature on the decision-making process of the model.")

    index = st.number_input("Enter the index of the observation to explain", min_value=0, max_value=len(X_test)-1, step=1, value=0)
    shap_decision_plot(rf_model, X_test, y_test, index, class_names)
    st.write("SHAP Decision Plot Overview:")
    st.write("- Baseline (Expected Value): The starting point of the plot, which is the average model output across all predictions (i.e., the expected value).")
    st.write("- Feature Contributions: How each feature's SHAP value moves the prediction from the baseline to the final prediction. Each feature's contribution is shown as a step in the plot.")
    st.write("- Final Prediction: The final model prediction for the specific instance after all feature contributions have been added to the baseline.")
    
elif selection == "Obesity Estimation Tool":
    st.header("Obesity Estimation Tool using Eating and Lifestyle habits")
    st.write("Take the below quiz and analyse your risk of obesity")
    
    # Define attribute inputs
    user_data = {}
    user_data['Gender'] = st.selectbox("Select Gender", options=["Female", "Male"])
    user_data['Age'] = st.slider("Select Age", min_value=0, max_value=100, value=25)
    user_data['Height'] = st.slider("Select Height (in meters)", min_value=0.5, max_value=2.5, value=1.75)
    user_data['Weight'] = st.slider("Select Weight (in kg)", min_value=20, max_value=200, value=70)
    user_data['family_history_with_overweight'] = st.radio("Family History with Overweight?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    user_data['FAVC'] = st.radio("Do you eat high caloric food frequently? 🍔", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    fcvc_options = {"Never": 1, "Sometimes": 2, "Always": 3}
    fcvc_selection = st.selectbox("Do you usually eat vegetables in your meals?🥕", options=list(fcvc_options.keys()))
    user_data['FCVC'] = fcvc_options[fcvc_selection]

    user_data['NCP'] = st.slider("How many main meals do you have daily?", min_value=1, max_value=6, value=3)
    user_data['CAEC'] = st.selectbox("Do you eat any food between meals?", options=["no", "Sometimes", "Frequently", "Always"])
    user_data['SMOKE'] = st.radio("Do you smoke? 🚬", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    user_data['CH2O'] = st.slider("How much water do you drink daily? (In Litres🥛)", min_value=1, max_value=3, value=2)
    user_data['SCC'] = st.radio("Do you monitor the calories you eat daily? ⏱️", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    user_data['FAF'] = st.slider("How often do you have physical activity? ⚽", min_value=0, max_value=7, value=2)
    user_data['TUE'] = st.slider("How many hours do you use technological devices daily? 📱", min_value=0, max_value=24, value=4)
    user_data['CALC'] = st.selectbox("How often do you drink alcohol?", options=["no", "Sometimes", "Frequently", "Always"])
    user_data['MTRANS'] = st.selectbox("Which transportation do you usually use?", options=["Walking 🚶‍♂️", "Bike 🚴🏻", "Motorbike 🏍️", "Automobile 🚗", "Public_Transportation 🚌"])
    
    if st.button("Predict"):
        user_df = pd.DataFrame(user_data, index=[0])
        
        # Ensure the user input data is properly encoded and scaled
        user_df['family_history_with_overweight'] = user_df['family_history_with_overweight'].map({0: 0, 1: 1})
        user_df['FAVC'] = user_df['FAVC'].map({0: 0, 1: 1})
        user_df['SMOKE'] = user_df['SMOKE'].map({0: 0, 1: 1})
        user_df['SCC'] = user_df['SCC'].map({0: 0, 1: 1})
        
        #user_df = pd.get_dummies(user_df, columns=['Gender', 'CAEC', 'CALC', 'MTRANS'])
        categorical_columns = ['Gender', 'CAEC', 'CALC', 'MTRANS']
        le = LabelEncoder()
        for column in categorical_columns:
            user_df[column] = le.fit_transform(user_df[column])

        
        # Add missing columns
        for col in X_train.columns:
            if col not in user_df.columns:
                user_df[col] = 0
        
        # Reorder columns to match training data
        user_df = user_df[X_train.columns]
        
        # Calculate BMI
        user_df['BMI'] = user_df['Weight'] / (user_df['Height'] ** 2)

        
        
        
        
        prediction = rf_model.predict(user_df)
        st.warning(f"The predicted obesity level is: {prediction[0]}")
        classes = ['Overweight_Level_I', 'Overweight_Level_II',
               'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
               'Obesity_Type_III']
        if prediction[0] in classes:
            st.warning("🔔 Please consult with a healthcare professional for a detailed analysis.")


elif selection == "Personalized Recommendations":
    st.header("Personalized Recommendations based on Obesity Level")
    st.divider()

    st.subheader("Exercise🏋 and Diet Recommendations🍉")

    recommendations = {
        "Insufficient_Weight": {
            "Exercise": [
                "Focus on strength training exercises to build muscle mass💪.",
                "Include compound movements like squats, deadlifts, and bench presses🏋.",
                "Aim for 3-4 days of strength training per week📆."
            ],
            "Diet": [
                "Increase your caloric intake with nutrient-dense foods🥗.",
                "Include plenty of proteins🍗, healthy fats, and complex carbohydrates🍙.",
                "Eat more frequently, aiming for 5-6 meals per day🍽."
            ]
        },
        "Normal_Weight": {
            "Exercise": [
                "Maintain a balanced workout routine with a mix of cardio🏃 and strength training🏋.",
                "Include activities like jogging🏃, cycling🚴, and weight lifting🏋.",
                "Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week📆."
            ],
            "Diet": [
                "Continue with a balanced diet rich in fruits🍎, vegetables🥕, lean proteins🥩, and whole grains🌾.",
                "Monitor portion sizes to maintain your weight.",
                "Stay hydrated🥛 and limit sugary drinks🚫."
            ]
        },
        "Overweight_Level_I": {
            "Exercise": [
                "Increase physical activity with more emphasis on cardio exercises🏃.",
                "Include activities like brisk walking🚶, swimming🏊, and aerobic classes🤸.",
                "Aim for at least 300 minutes of moderate-intensity activity per week."
            ],
            "Diet": [
                "Reduce caloric intake focusing on portion control.",
                "Choose low-calorie, nutrient-dense foods like vegetables🥕 and lean proteins🥩.",
                "Avoid sugary snacks and beverages🚫."
            ]
        },
        "Overweight_Level_II": {
            "Exercise": [
                "Focus on regular cardio workouts combined with some strength training🏋.",
                "Incorporate exercises like running🏃, cycling🚴, and circuit training.",
                "Aim for 300 minutes of moderate-intensity exercise per week."
            ],
            "Diet": [
                "Adopt a calorie-restricted diet with an emphasis on healthy eating.",
                "Increase the intake of fruits🍎, vegetables🥕, and whole grains🌾.",
                "Limit the consumption of high-calorie, sugary, and fatty foods🚫."
            ]
        },
        "Obesity_Type_I": {
            "Exercise": [
                "Engage in daily physical activities, prioritizing low-impact cardio exercises🏃.",
                "Include activities like swimming🏊, walking🚶, and using an elliptical machine.",
                "Aim for at least 300 minutes of moderate-intensity activity per week, increasing intensity gradually📆."
            ],
            "Diet": [
                "Follow a structured, calorie-controlled diet plan.",
                "Increase the consumption of fiber-rich foods like vegetables🥦 and whole grains🌽.",
                "Consult with a dietician to create a personalized meal plan."
            ]
        },
        "Obesity_Type_II": {
            "Exercise": [
                "Participate in daily moderate-intensity cardio activities🏃.",
                "Include exercises such as brisk walking🚶, water aerobics🏊, and stationary cycling🚴.",
                "Work towards achieving at least 300 minutes of activity per week📆."
            ],
            "Diet": [
                "Adhere to a strict, low-calorie diet supervised by a healthcare provider🧑‍⚕️.",
                "Focus on nutrient-dense foods that are low in calories.",
                "Avoid high-sugar and high-fat foods completely🚫."
            ]
        },
        "Obesity_Type_III": {
            "Exercise": [
                "Incorporate gentle, low-impact exercises and gradually increase intensity📈.",
                "Consider activities like walking🚶, water aerobics🏊, and chair exercises.",
                "Aim for a gradual increase in activity levels under professional supervision🧑‍⚕️."
            ],
            "Diet": [
                "Follow a medically-supervised, very low-calorie diet if recommended.",
                "Prioritize high-protein, low-carbohydrate foods to aid weight loss.",
                "Regularly consult with healthcare providers for diet adjustments🧑‍⚕️."
            ]
        }
    }

    predicted_class = st.selectbox("Select Predicted Obesity Class:", options=list(recommendations.keys()))

    st.subheader(f"Exercise Recommendations for {predicted_class.replace('_', ' ')}")
    for exercise in recommendations[predicted_class]["Exercise"]:
        st.write(f"- {exercise}")

    st.subheader(f"Diet Recommendations for {predicted_class.replace('_', ' ')}")
    for diet in recommendations[predicted_class]["Diet"]:
        st.write(f"- {diet}")

elif selection == "User Feedback":
    st.markdown('We value your inputs and we would appreciate if you could give us some feedback to improve')
    feedback = st.text_area("Your feedback")

    # Dropdown for rating
    rating = st.selectbox("Rate our tool", ["Select a rating", "Excellent", "Good", "Average", "Poor"])

    # Button to submit feedback
    if st.button("Send"):
        if not feedback:
            st.warning("Please enter your feedback before sending🙏.")
        elif rating == "Select a rating":
            st.warning("Please select a rating before sending🙏.")
        else:
            # Simulate sending feedback
            # In a real application, you would send the feedback here
            st.success("Thank you for your feedback and rating! 😊")
