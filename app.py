
#Install Streamlit
#!pip install streamlit

#Write Basic Code for the App
import streamlit as st

# Title of the app
st.title("My First Streamlit App")

# Adding a header and some text
st.header("Hello, Streamlit!")
st.write("This is a simple Streamlit application.")

# Create an interactive widget (slider)
age = st.slider("Select your age", 0, 100, 25)
st.write("Your selected age is:", age)

#Add More Features
name = st.text_input("Enter your name", "Type here...")
st.write("Hello,", name)

#Checkbox that show content based on user input
if st.checkbox("Show text"):
    st.write("Hello, Streamlit!")

    #Graphs and Plots:
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)

print('test')