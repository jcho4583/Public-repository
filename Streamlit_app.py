import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Airline passenger satisfaction analysis", page_icon="✈️")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!", "Conclusions"])

# Load dataset
df = pd.read_csv('data/cleaned_airline_passenger_satisfaction.csv')
#df = df.drop(columns = "Unnamed: 0", inplace = True)

# Home Page
if page == "Home":
    st.title("📊 Airline passenger satisfaction Dataset Explorer")
    st.subheader("Welcome to our Airline passenger sastisfaction analysis dataset explorer app!")
    st.write("""
        This app provides an interactive platform to explore the Airline passenger sastisfaction dataset. 
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        Use the sidebar to navigate through the sections.
    """)
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Korean_Air_Airbus_A380-861%3B_HL7612%40HKG%3B04.08.2011_615dq_%286207233991%29.jpg/960px-Korean_Air_Airbus_A380-861%3B_HL7612%40HKG%3B04.08.2011_615dq_%286207233991%29.jpg', caption="Airbus A380")
    st.write("Use the sidebar to navigate between different sections.")


# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("""
        This dataset contains an airline passenger satisfaction survey.  It contains the responses from 103,904 respondents regarding satisfaction
        levels on various metrics, as well as metrics related to the airline's performance.  The goal is determine if a passenger will be satisfied
        or dissatisfied based on these metrics.
    """)
  #  st.image('https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png', caption="Iris Dataset")

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots'])

    obj_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by satisfaction"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='satisfaction', y=b_selected_col, title=chart_title, color='satisfaction'))

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))

#    if 'Count Plots' in eda_type:
#        st.subheader("Count Plots - Visualizing Categorical Distributions")
#        selected_col = st.selectbox("Select a categorical variable:", obj_cols)
#        if selected_col:
#            chart_title = f'Distribution of {selected_col.title()}'
#            st.plotly_chart(px.histogram(df, x=selected_col, color='satisfaction', title=chart_title))

# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = ["Gate location", "Gender", "Departure/Arrival time convenient", "satisfaction"])
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=35, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

# Make Predictions Page
elif page == "Make Predictions!":
    st.title("✈️ Make Predictions")

    st.subheader("Adjust the values below to make predictions on the Iris dataset:")

    # User inputs for prediction
#   Gender = st.slider("Gender", min_value=0, max_value=1, value=1)
    Customer_Type = st.slider("Customer Type", min_value=0, max_value=1, value=1)
    Age = st.slider("Age", min_value=0, max_value=100, value=50)
    Type_of_Travel = st.slider("Type of Travel", min_value=0, max_value=1, value=1)
    Class = st.slider("Class", min_value=0, max_value=2, value=0)
    Flight_Distance = st.slider("Flight Distance", min_value=0, max_value=10000, value=2000)
    Inflight_wifi_service = st.slider("Inflight wifi service", min_value=0, max_value=5, value=3)
#    Gender = st.slider("Gender", min_value=0, max_value=0, value=1)
    Ease_of_Online_booking = st.slider("Ease of Online booking", min_value=0, max_value=5, value=3)
#    Gate_location = st.slider("Gate location", min_value=0, max_value=0, value=1)
    Food_and_drink = st.slider("Food and drink", min_value=0, max_value=5, value=3)
    Online_boarding = st.slider("Online boarding", min_value=0, max_value=5, value=3)
    Seat_comfort = st.slider("Seat comfort", min_value=0, max_value=5, value=3)
    Inflight_entertainment = st.slider("Inflight entertainment", min_value=0, max_value=5, value=3)
    On_board_service = st.slider("On-board service", min_value=0, max_value=5, value=3)
    Leg_room_service = st.slider("Leg room service", min_value=0, max_value=5, value=3)
    Baggage_handling = st.slider("Baggage handling", min_value=0, max_value=5, value=3)
    Checkin_service = st.slider("Checkin service", min_value=0, max_value=5, value=3)
    Inflight_service = st.slider("Inflight service", min_value=0, max_value=5, value=3)
    Cleanliness = st.slider("Cleanliness", min_value=0, max_value=5, value=3)
    Departure_Delay_in_Minutes = st.slider("Departure Delay in Minutes", min_value=0, max_value=1000, value=0)
    Arrival_Delay_in_Minutes = st.slider("Arrival Delay in Minutes", min_value=0, max_value=1000, value=0)
    
#Departure/Arrival time convenient	Inflight service	Cleanliness	Departure Delay in Minutes	Arrival Delay in Minutes

    # User input dataframe
    user_input = pd.DataFrame({
 #       "Gender": [Gender],
        "Customer Type": [Customer_Type],
        "Age": [Age],
        "Type of Travel": [Type_of_Travel],
        "Class": [Class],
        "Flight Distance": [Flight_Distance],
        "Inflight wifi service": [Inflight_wifi_service],
#        "Inflight wifi service": [Inflight_wifi_service],
        "Ease of Online booking": [Ease_of_Online_booking],
#        "Gate location": [Gate_location],
        "Food and drink": [Food_and_drink],
        "Online boarding": [Online_boarding],
        "Seat comfort": [Seat_comfort],
        "Inflight entertainment": [Inflight_entertainment],
        "On-board service": [On_board_service],
        "Leg room service": [Leg_room_service],
        "Baggage handling": [Baggage_handling],
        "Checkin service": [Checkin_service],
        "Inflight service": [Inflight_service],
        "Cleanliness": [Cleanliness],
        "Departure Delay in Minutes": [Departure_Delay_in_Minutes],
        "Arrival Delay in Minutes": [Arrival_Delay_in_Minutes]
    })

    st.write("### Your Input Values")
    st.dataframe(user_input)

    # Use RandomForestClassifier as the model for predictions
    model = RandomForestClassifier()
    X = df.drop(columns = ['satisfaction', 'Gender', 'Departure/Arrival time convenient', 'Gate location'])
    y = df['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]

    # Display the result
    st.write(f"The model's customer satisfaction prediction: **{prediction}**")
    st.balloons()

if page == "Conclusions":

    st.title("Comparing KNN vs Logistic Regression vs Random Forest")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("KNN (K = 3)")
        st.write("Test Accuracy = 93%")

    with col2:
        st.header("Logistic Regression")
        st.write("Test Accuracy = 87%")

    with col3:
        st.header("Random Forest")
        st.write("Test Accuracy = 96%")
    
    st.divider()

    st.title("Which model is best?")
    st.write("Against a baseline of 57%, all three models offer significant improvement over the baseline." \
    "  However, the Random Forest model performed at best with 96% accuracy.")