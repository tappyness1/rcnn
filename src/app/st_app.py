import os
import sys

import cv2
import hydra
import plotly.express as px
import streamlit as st
from omegaconf import DictConfig, OmegaConf

sys.path.append("./")
import json

import requests


def show_image(folder, img_fname):
    img = cv2.imread(f"{folder}{img_fname}")
    fig = px.imshow(img[..., ::-1], aspect="equal")
    return fig


@hydra.main(version_base=None, config_path="../../conf", config_name="cfg")
def app(cfg: DictConfig):
    """ """
    st.markdown(
        """ <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style> """,
        unsafe_allow_html=True,
    )
    # st.write(f"Current dir's folders/files: {os.listdir('./')}")
    path = cfg["app"]["folder_path"]  # TODO: move to config
    all_files = os.listdir(path)
    choice = st.sidebar.selectbox(label="All Images", options=all_files)
    image_path = {"fname": f"{path}{choice}"}
    # if we have multiple requests, consider json
    res = requests.post(
        url="http://127.0.0.1:8000/predict", data=json.dumps(image_path)
    )

    st.plotly_chart(show_image(path, choice), use_container_width=True)
    st.write(f"Prediction: {res.text}")


if __name__ == "__main__":
    app()
