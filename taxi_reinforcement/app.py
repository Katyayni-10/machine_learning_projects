import streamlit as st
import gymnasium as gym
import numpy as np
import pickle
import pandas as pd
import time

st.set_page_config(
    page_title="Taxi RL Dashboard",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("q_table.pkl", "rb") as file:
        q_table = pickle.load(file)
    return q_table

q_table = load_model()

st.markdown("""
<style>

.main{
background-color:#0E1117;
color:white;
}

.title{
font-size:45px;
font-weight:bold;
text-align:center;
color:#FFD700;
}

.subtitle{
text-align:center;
font-size:20px;
color:#DDDDDD;
}

.metric{
background:#1B263B;
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:15px;
}

.footer{
text-align:center;
color:gray;
padding-top:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Taxi Reinforcement Learning Dashboard</div>", unsafe_allow_html=True)

# st.markdown("<div class='subtitle'>Q-Learning using Gymnasium Taxi-v3 Environment</div>", unsafe_allow_html=True)

# st.divider()

# st.sidebar.title("Project Details")

# st.sidebar.write("Algorithm : Q-Learning")

# st.sidebar.write("Environment : Taxi-v3")

# st.sidebar.write("States : 500")

# st.sidebar.write("Actions : 6")

# st.sidebar.success("Model Loaded Successfully")


col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric("States",500)

with col2:
    st.metric("Actions",6)

with col3:
    st.metric("Q Table Rows",q_table.shape[0])

with col4:
    st.metric("Q Table Columns",q_table.shape[1])

st.divider()

actions={
0:"⬇ Move South",
1:"⬆ Move North",
2:"➡ Move East",
3:"⬅ Move West",
4:"Pickup Passenger",
5:"Drop Passenger"
}

if st.button("Run Taxi Simulation",use_container_width=True):

    env=gym.make("Taxi-v4")

    state,info=env.reset()

    total_reward=0

    done=False

    step=1

    history=[]

    progress=st.progress(0)

    status=st.empty()

    table=st.empty()

    while not done:

        action=np.argmax(q_table[state])

        next_state,reward,terminated,truncated,info=env.step(action)

        done=terminated or truncated

        history.append({
            "Step":step,
            "Action":actions[action],
            "Reward":reward
        })

        total_reward+=reward

        progress.progress(min(step,100))

        status.info(
            f"""
Step : {step}

Action : {actions[action]}

Reward : {reward}

Total Reward : {total_reward}
"""
        )

        table.dataframe(pd.DataFrame(history),
                        use_container_width=True)

        state=next_state

        step+=1

        time.sleep(0.2)

    progress.progress(100)

    st.success("Passenger Reached Successfully")

    st.metric("Final Reward",total_reward)

st.divider()

# st.markdown(
# """
# ### About this Project

# This project demonstrates **Reinforcement Learning**
# using the **Taxi-v3** environment from Gymnasium.

# The taxi learns how to:

# - Find the passenger
# - Pick up the passenger
# - Reach destination
# - Drop passenger successfully

# using the **Q-Learning algorithm**.
# """
# )

# st.markdown("<div class='footer'>Made with ❤️ using Streamlit & Gymnasium</div>",unsafe_allow_html=True)