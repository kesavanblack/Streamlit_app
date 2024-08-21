import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle  # to load a saved model
import base64  # to open .gif files in Streamlit app
import seaborn as sns
import numpy as np

@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val, None)

def get_value(val, my_dict):
    return my_dict.get(val, None)

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])

if app_mode == 'Home':
    # Title for the home page
    st.title('LOAN PREDICTION')

    # Display an image (ensure the image file is in the correct path)
    st.image('4k.jpg')

    # Display the dataset
    st.markdown('### Dataset:')
    data = pd.read_csv('Player Batting Stats - Most Runs.csv')
    st.write(data.head())

    # Bar Chart
    st.markdown('### Team vs Runs - Bar Chart')
    st.bar_chart(data[['Team', 'Runs']].head(20))

    # Line Chart
    st.markdown('### Team vs Runs - Line Chart')
    st.line_chart(data[['Team', 'Runs']].head(20))
    
    # Scatter Plot
    st.markdown('### Team vs Runs - Scatter Plot')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Team', y='Runs', data=data)
    plt.xlabel('Team')
    plt.ylabel('Runs')
    st.pyplot(plt)

    # Pie Chart
    st.markdown('### Team vs Runs - Pie Chart')
    plt.figure(figsize=(8, 8))
    plt.pie(data['Runs'].head(10), labels=data['Team'].head(10), autopct='%1.1f%%')
    st.pyplot(plt)

elif app_mode == 'Prediction':
    # Display an image and subheader
    st.image('4k.jpg')
    st.subheader("Sir/Mme, YOU need to fill in all necessary information to get a reply to your loan request!")

    # Sidebar for client information
    st.sidebar.header("Information about the client:")

    # Dictionaries for mapping input to numerical values
    gender_dict = {"Male": 1, "Female": 2}
    feature_dict = {"No": 1, "Yes": 2}
    edu = {'Graduate': 1, 'Not Graduate': 2}
    prop = {'Rural': 1, 'Urban': 2, 'Semiurban': 3}

    # Sidebar sliders and inputs
    ApplicantIncome = st.sidebar.slider('Applicant Income', 0, 10000, 0)
    CoapplicantIncome = st.sidebar.slider('Coapplicant Income', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('Loan Amount in K$', 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox('Loan Amount Term', (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History = st.sidebar.radio('Credit History', (0.0, 1.0))
    Gender = st.sidebar.radio('Gender', tuple(gender_dict.keys()))
    Married = st.sidebar.radio('Married', tuple(feature_dict.keys()))
    Self_Employed = st.sidebar.radio('Self Employed', tuple(feature_dict.keys()))
    Dependents = st.sidebar.radio('Dependents', options=['0', '1', '2', '3+'])
    Education = st.sidebar.radio('Education', tuple(edu.keys()))
    Property_Area = st.sidebar.radio('Property Area', tuple(prop.keys()))

    # Mapping Dependents to one-hot encoding
    class_0, class_1, class_2, class_3 = 0, 0, 0, 0
    if Dependents == '0':
        class_0 = 1
    elif Dependents == '1':
        class_1 = 1
    elif Dependents == '2':
        class_2 = 1
    else:  # '3+'
        class_3 = 1

    # Mapping Property_Area to one-hot encoding
    Rural, Urban, Semiurban = 0, 0, 0
    if Property_Area == 'Urban':
        Urban = 1
    elif Property_Area == 'Semiurban':
        Semiurban = 1
    else:  # 'Rural'
        Rural = 1

    # Collect all input data into a dictionary
    input_data = {
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Gender": gender_dict[Gender],
        "Married": feature_dict[Married],
        "Self_Employed": feature_dict[Self_Employed],
        "Education": edu[Education],
        "Dependents_0": class_0,
        "Dependents_1": class_1,
        "Dependents_2": class_2,
        "Dependents_3+": class_3,
        "Property_Area_Rural": Rural,
        "Property_Area_Urban": Urban,
        "Property_Area_Semiurban": Semiurban,
    }

    # Display input data for prediction
    st.write("Input Data for Prediction:", input_data)

    if st.button("Predict"):
        try:
            # Load the model
            loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))
            
            # Convert the input data to the appropriate format
            single_sample = np.array(list(input_data.values())).reshape(1, -1)
            
            # Make the prediction
            prediction = loaded_model.predict(single_sample)
            
            # Show results based on prediction
            if prediction[0] == 0:
                st.error('According to our Calculations, you will not get the loan from Bank')
                with open("green-cola-no.gif", "rb") as file:
                    data_url_no = base64.b64encode(file.read()).decode("utf-8")
                st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="no loan gif">', unsafe_allow_html=True)
            else:
                st.success('Congratulations!! you will get the loan from Bank')
                with open("6m-rain.gif", "rb") as file_:
                    data_url = base64.b64encode(file_.read()).decode("utf-8")
                st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="loan approved gif">', unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Model file 'Random_Forest.sav' not found. Please ensure the file is in the correct directory.")
        except Exception as e:
            st.error(f"An error occurred during the prediction: {e}")
