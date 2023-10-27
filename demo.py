import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import pickle
import numpy as np
from numpy import argmax
import jajapy as ja
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings('ignore')


st.set_page_config(page_title="Solar Irradinace Forcasting!!!", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Solar Irradinace Forcasting")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    os.chdir(r"V:/sem 5/ST/ST Project/")
    df = pd.read_csv("1.csv", encoding = "ISO-8859-1")
st.write("""Upload File""")
st.dataframe(df)
def training_test_set(cols, len_seq=14):
    arr = np.loadtxt('V:/sem 5/ST/ST Project/1.csv', delimiter=',', dtype=str)
    nb_distributions = len(cols)
    arr = arr[:, cols]
    print(arr[0])

    training_set = np.array(arr[1:251], dtype=np.float64)
    test_set = np.array(arr[251:325], dtype=np.float64)

    complete_seq = len(training_set) // len_seq
    training_set = training_set[:len(training_set) - len(training_set) % len_seq]
    training_set = training_set.reshape((complete_seq, len_seq, len(cols)))

    complete_seq = len(test_set) // len_seq
    test_set = test_set[:len(test_set) - len(test_set) % len_seq]
    test_set = test_set.reshape((complete_seq, len_seq, len(cols)))

    # Ensure sequences is a list of arrays
    training_sequences = [np.array(seq) for seq in training_set]
    test_sequences = [np.array(seq) for seq in test_set]

    training_set = ja.Set(training_sequences, np.ones(len(training_sequences)), t=3)
    test_set = ja.Set(test_sequences, np.ones(len(test_sequences)), t=3)


    return training_set, test_set



def testing(m, seq, steps=5):
    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
    fig.suptitle("Solar Irradiance Forecasting", fontsize=16)
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Adjust spacing between subplots

    alphas = m._initAlphaMatrix(steps)
    alphas = m._updateAlphaMatrix(seq[:steps], 0, alphas)
    alphas = alphas[-1]
    current = np.argmax(alphas)

    forecast = m.run(len(seq) - steps, current)

    features = ["cloud_cover (%)", "wind_speed (m/s)", "wind_gust (m/s)", "humidity (%)", "pressure (hPa)", "temp_mean (°C)", "temp_min (°C)", "temp_max (°C)"]

    for i in range(8):
        row = i // 4
        col = i % 4
        x_values = list(range(1, len(seq) + 1))
        y_values_actual = [seq[j][i] for j in range(len(seq))]
        y_values_forecast = [seq[steps - 1][i]] + [forecast[j][i] for j in range(len(forecast))]

        axs[row, col].plot(x_values, y_values_actual, label='Actual', color='blue')
        axs[row, col].plot(x_values[steps - 1:], y_values_forecast, label='Forecast', color='orange')
        axs[row, col].set_xlabel("Time (steps)")
        axs[row, col].set_ylabel(features[i] + " (values)")
        axs[row, col].legend()

    # Hide empty subplots
    for i in range(2):
        for j in range(4):
            if (i * 4 + j) >= 8:
                axs[i, j].axis('off')

    st.pyplot(fig)

# ... (rest of your code)

    



def example_8():

	
    training_set, test_set = training_test_set([19,20,21,22,23,27,28,29], 7)
    print(training_set)
    print("Number of sequences in training set:", len(training_set.sequences))


    
    for seq in training_set.sequences:
         print(seq)

    print("First sequence in the training set:\n",training_set.sequences[0])
	
    
    
    initial_hypothesis = ja.GoHMM_random(nb_states=15, nb_distributions=8,
				      					 min_mu=-5.0,max_mu=5.0,
										 min_sigma=1.0,max_sigma=5.0)
    output_model = ja.BW().fit(training_set, initial_hypothesis)
    for hour in range(1,3,1):
        testing(output_model,test_set.sequences[hour],steps=5)
    with open('m_pickle','wb') as f:
         pickle.dump(output_model,f)


mpl.rcParams['figure.figsize'] = (20,10)


if __name__ == '__main__':
	example_8()
