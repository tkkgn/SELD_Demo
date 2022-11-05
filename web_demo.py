import tempfile
import streamlit as st
from pathlib import Path
from pydub import AudioSegment

from Scatter_Animation import Animation

import librosa
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')

# Import everything needed to edit video clips
from moviepy.editor import *


import numpy as np
import torch
import pandas as pd

from models.MMUB import MIMO_UNet_Beamforming
from models.SELDNet import Seldnet_augmented
from utility_functions import (gen_submission_list_task2, load_model,
                               spectrum_fast)

import storage_params as sp


class Predictor:
    def setup(self):
        """Load the model"""
        use_cuda = False
        gpu_id = 0
        if use_cuda:
            self.device = "cuda:" + str(gpu_id)
        else:
            self.device = "cpu"
        task2_pretrained_path = "RESULTS/Task2/baseline_task2_checkpoint"
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

def animation_directive(df):
        #st.write(len(df))
    data = np.column_stack((df["Frames"].to_numpy(),df["X"].to_numpy(),df["Y"].to_numpy()))
    #st.write(data)
    name = df["Class Name"].to_numpy()
    #display = Animation(data,name).get_animation()
    #st.write("done")
    
    fig, ax = plt.subplots()
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    ax.grid()
    
    current = 0
    frame =0
    
    for temp in range(len(name)):
        ax.set_title("the frame number: {}".format(frame))
        if frame == int(data[temp,0]):
            continue
        xtemp=data[current:temp,1]
        ytemp=data[current:temp,2]
        nameT = name[current:temp]
        print(frame,xtemp,ytemp,nameT)
        
        center = ax.scatter(0,0,color = "r")
        bha= ax.scatter(xtemp,ytemp,s=100)
        ann_list = []
        for i in range(len(nameT)):
            ann_list.append(ax.annotate(nameT[i],(xtemp[i],ytemp[i])))
            
        plt.draw(fig)
        plt.pause(0.1)
        bha.remove()
        for i, a in enumerate(ann_list):
            a.remove()        
        current=temp
        frame +=1
        if frame == 300:
            break
    plt.close() 



def merge_animation_to_audiofile(anim_path,audio_path):

    result_path = os.path.join(sp.dir_path,sp.result_file)
    # loading video dsa gfg intro video
    clip = VideoFileClip(anim_path)
    
    # getting only first 5 seconds
    clip = clip.subclip(0, 30)
    
    # loading audio file
    audioclip = AudioFileClip(audio_path).subclip(0, 30)
    
    # adding audio to the video clip
    videoclip = clip.set_audio(audioclip)
    
    videoclip.write_videofile(result_path)
    return result_path

    # showing video clip
    #videoclip.ipython_display()


def animation_with_matplot_FuncAnimation(df,audio_path):
    #convert dataframe to numpy for plotting animation
    data = np.column_stack((df["Frames"].to_numpy(),df["X"].to_numpy(),df["Y"].to_numpy()))
    name = df["Class Name"].to_numpy()
    
    with st.spinner('Wait for loading animation...'):
        anim = Animation(data,name)
        result_path = merge_animation_to_audiofile(anim.VIDEO_PATH,audio_path)
    st.success('Done!')
    
       
    #read video
    video_file = open(result_path, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    

def add_blank_rows(df):
    i=0
    cur_frame = -1
    frames = 300
    df_new = pd.DataFrame(columns=df.columns)
    frame_list = df['Frames'].values.tolist()
    # st.write(len(frame_list))
    # print(cur_frame in frame_list)
    # quit()
    # temp = [i,-1,"",0,0,0]
    # print(temp)
    # #print(df['Frames'][indx])
    # quit()
    while True:
        try:
            if cur_frame == frame_list[i]:
                df_new = pd.concat([df_new,df.iloc[[i]]],
                                    ignore_index=True,
                                    sort = False)
                i += 1
            elif cur_frame < frame_list[i]:
                if cur_frame + 1 == frame_list[i]:
                    cur_frame += 1
                    continue
                
                cur_frame += 1
                if cur_frame == frames:
                    break
                #add null row
                temp = [cur_frame,-1,"",0,0,0]
                df_new = pd.concat([df_new,pd.DataFrame([temp],columns=df.columns)],
                                ignore_index=True,
                                sort = False)
                
                
        except:
            cur_frame += 1
            if cur_frame == frames:
                break
            temp = [cur_frame,-1,"",0,0,0]
            df_new = pd.concat([df_new,pd.DataFrame([temp],columns=df.columns)],
                            ignore_index=True,
                            sort = False)
            
          
    return df_new
    
    


def visual_dataframe(np_arr,option):
    name = {
        0: 'Chink and clink',
        1: 'Computer keyboard',
        2: 'cupboard open/close',
        3: 'Drawer open/close',
        4: 'Female/woman speaking',
        5: 'Finger snapping',
        6: 'Key Jangling',
        7: 'Knock',
        8: 'Laughter',
        9: 'Male/woman speaking',
        10: 'Printer',
        11: 'Scissors',
        12: 'Telephone',
        13: 'Writing'
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
    
    df = add_blank_rows(df)
    st.dataframe(df)
    return df
    #df.to_csv('test.csv')

def save_fileinput(file_input):
    fpath = os.path.join(sp.dir_path,sp.raw_audio)
    file_convert = AudioSegment.from_wav(file_input)
    file_convert.export(fpath,format='wav')

    st.markdown(f"""
    ### Files
    - {fpath}
    """,
    unsafe_allow_html=True) # display file name
    return fpath




if __name__ == '__main__':
    st.title('SELD')
    # run model
    demo = Predictor()
    demo.setup()
    # process
    col1, col2 = st.columns(2)
    result = st.container()
    with col1:
        data = st.selectbox(
            'data type?',
            ('L3DAS22 Dataset(default)', 'DCASE2022 Dataset', 'Mp4'))
    with col2:
        file_input = st.file_uploader("Input file")


    with result:
        if file_input is not None:
            #try:
                #Save file_input to audio path which facilitate merging audio with animation 
                apath = None
                option = 0
                match data:
                    case 'L3DAS22 Dataset(default)':
                        st.audio(file_input)
                        output = demo.predict(file_input)
                        apath = save_fileinput(file_input)
                        option = 1
                    case 'DCASE2022 Dataset':
                        st.audio(file_input)
                        processed_input = cut_audio(file_input)
                        output = demo.predict(processed_input)
                        apath = save_fileinput(file_input)
                        option = 2
                    case 'Mp4':
                        st.video(file_input)
                        st.write('Building soon')
                        # processed_input = mp4_2_wav(file_input)
                        # Buiding
                df = visual_dataframe(output,option)
                if apath is not None:
                    animation_with_matplot_FuncAnimation(df,apath)
            #except:
            #    st.write('check input')
