import tempfile
import streamlit as st
from pathlib import Path
from pydub import AudioSegment

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from models.MMUB import MIMO_UNet_Beamforming
from models.SELDNet import Seldnet_augmented
from utility_functions import (gen_submission_list_task2, load_model,
                               spectrum_fast)


class Predictor:
    def setup(self):
        """Load the model"""
        use_cuda = False
        gpu_id = 0
        if use_cuda:
            self.device = "cuda:" + str(gpu_id)
        else:
            self.device = "cpu"
        task2_pretrained_path = "RESULTS/Task2/checkpoint"
        self.model_task2 = Seldnet_augmented(
            time_dim=2400,
            freq_dim=256,
            input_channels=4,
            output_classes=14,
            pool_size=[[8, 2], [8, 2], [2, 2], [1, 1]],
            pool_time=True,
            rnn_size=256,
            n_rnn=3,
            fc_size=1024,
            dropout_perc=0.3,
            cnn_filters=[64, 128, 256, 512],
            class_overlaps=3,
            verbose=False,
        )
        self.model_task2 = self.model_task2.to(self.device)
        _ = load_model(self.model_task2, None, task2_pretrained_path, use_cuda)
        self.model_task2.eval()

    def predict(self, input):
        """Compute prediction"""
        sr_task2 = 32000
        x, sr = librosa.load(input, sr=sr_task2, mono=False)
        x = spectrum_fast(
            x, nperseg=512, noverlap=112, window="hamming", output_phase=False
        )
        x = torch.tensor(x).float().unsqueeze(0)
        with torch.no_grad():
            sed, doa = self.model_task2(x)
        sed = sed.cpu().numpy().squeeze()
        doa = doa.cpu().numpy().squeeze()

        # write output
        seld = gen_submission_list_task2(
            sed, doa, max_overlaps=3, max_loc_value=1
        )
        #output_path_png = Path(tempfile.mkdtemp()) / "output.png"
        #plot_task2(output_path_png, sed, doa)
        return seld


def plot_task2(output_path, sed, doa):
    n = sed.shape[1]
    x = doa[:, :n]
    y = doa[:, n: n * 2]
    z = doa[:, n * 2:]
    positions = np.arange(0, sed.shape[0] + 1, 50)
    labels = np.array(positions / 10, dtype="int32")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.subplot(221)
    plt.title("Sound activations")
    plt.pcolormesh(sed.T)
    plt.ylabel("Sound class")
    plt.xticks(positions, labels)

    plt.subplot(222)
    plt.title("X axis")
    plt.pcolormesh(x.T)
    plt.xticks(positions, labels)
    plt.subplot(223)
    plt.title("Y axis")
    plt.pcolormesh(y.T)
    plt.ylabel("Sound class")
    plt.xlabel("Time")
    plt.xticks(positions, labels)
    plt.subplot(224)
    plt.title("Z axis")
    plt.pcolormesh(z.T)
    plt.xticks(positions, labels)
    plt.xlabel("Time")

    plt.tight_layout()
    st.pyplot()
    plt.savefig(output_path, format="png", dpi=300)


def add_silence(audio):
    sec_segment = AudioSegment.silent(duration=30 * 1000)  # duration in milliseconds, add 30 seconds
    audio_tmp = AudioSegment.from_wav(audio)
    audio_output = audio_tmp + sec_segment
    return audio_output.export(format='wav')


def cut_audio(audio):
    audio_tmp = AudioSegment.from_wav(audio)
    first_30_seconds = audio_tmp[:30 * 1000]  # duration in milliseconds, cut first 30 seconds
    return first_30_seconds.export(format='wav')


def mp4_2_wav(video):
    mp4_version = AudioSegment.from_file(video, "mp4")
    first_30_seconds = mp4_version[:30 * 1000]
    return first_30_seconds.export(format='wav')


def visual_dataframe(np_arr):
    name = {
        0: 'computer keyboard',
        1: 'drawer open/close',
        2: 'cupboard open/close',
        3: 'finger snapping',
        4: 'keys jangling',
        5: 'knock ',
        6: 'laughter',
        7: 'scissors',
        8: 'telephone',
        9: 'writing',
        10: 'chink and clink',
        11: 'printer',
        12: 'female speech',
        13: 'male speech'
    }
    df = pd.DataFrame(np_arr, columns=['Frames', 'Class', 'X', 'Y', 'Z'])
    # 1.0000 --> 1
    df['Frames'] = df.apply(lambda row: int(row.Frames), axis=1)
    df['Class'] = df.apply(lambda row: int(row.Class), axis=1)
    # map class name
    df['Class Name'] = df['Class'].map(name)
    # col = ['Frames', 'Class', 'X', 'Y', 'Z','Class Name']  --> col = ['Frames', 'Class', 'Class Name', 'X', 'Y', 'Z']
    new_index_col = ['Frames', 'Class', 'Class Name', 'X', 'Y', 'Z']
    df = df[new_index_col]
    # df['Class Name'] = df.apply(lambda row: name[int(row.Class)], axis=1)
    st.dataframe(df)


if __name__ == '__main__':
    st.title('SELD')
    # run model
    demo = Predictor()
    demo.setup()
    # process
    col1, col2 = st.columns(2)
    with col1:
        data = st.selectbox(
            'data type?',
            ('L3DAS22 Dataset(default)', 'DCASE2022 Dataset', 'Mp4'))
    with col2:
        file_input = st.file_uploader("Input file")

    if file_input is None:
        st.stop()
    else:
        try:
            match data:
                case 'L3DAS22 Dataset(default)':
                    st.audio(file_input)
                    output = demo.predict(file_input)
                case 'DCASE2022 Dataset':
                    st.audio(file_input)
                    processed_input = cut_audio(file_input)
                    output = demo.predict(processed_input)
                case 'Mp4':
                    st.video(file_input)
                    st.write('Building soon')
                    # processed_input = mp4_2_wav(file_input)
                    # Buiding
            visual_dataframe(output)
        except:
            st.write('check input')
