import pickle
import sounddevice as sd
import scipy
from scipy.io.wavfile import write
import wavio as wv
from python_speech_features import mfcc
import numpy as np
import sys
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from pathlib import Path

pickled_model = pickle.load(open('finalized_model.sav', 'rb'))

def create_ceps(wavfile):
    sr, speech = scipy.io.wavfile.read(wavfile)
        #(sr)
    ceps=mfcc(speech)
        #ceps, mspec, spec= mfcc(song_array)
        #print(ceps.shape)

    bad = np.where(np.isnan(ceps))
    bad2=np.where(np.isinf(ceps))
    ceps[bad]=0
    ceps[bad2]=0
    write_ceps(ceps, wavfile)
def write_ceps(ceps, wavfile):
    base_wav, ext = os.path.splitext(wavfile)
    data_wav =  "running_test/" + base_wav.split("/")[1] + ".ceps"
    np.save(data_wav, ceps)

def read_ceps():
        X= []
        directory = "running_test/"
        print(os.listdir(directory))
        for filename in os.listdir(directory):
            #print(filename)
            if filename.endswith("npy"):
                ceps = np.load(directory + filename)
                num_ceps = len(ceps)
                X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))

        #print(np.array(X).shape)
        return np.array(X)

if len(sys.argv) == 1:
    import shutil
    path = 'running_test/'
    try:
       shutil.rmtree(path)
    except OSError as x:
      print("Error occured: %s : %s" % (path, x.strerror))

    os.mkdir(path)
    
    freq = 44100
    # Recording duration
    duration = 10
    recording = sd.rec(int(duration * freq), 
                    samplerate=freq, channels=1)
    print("RECORDING    ----------")

    # Record audio for the given number of seconds
    sd.wait()
    print("Stopped RECORDING    ----------")


    write("running_test/recording.wav", freq, recording)

    ceps =create_ceps("running_test/recording.wav")
    ceps_list = read_ceps()

    prediction = pickled_model.predict(ceps_list)
    print(prediction)
    if prediction[0] == 1:
        print("The model predicts that this is a human speaking!")

    else:
        print("Warning, the model predicts that this is an AI voice!")
    #import shutil
    #path = 'running_test/'
    #try:
     #  shutil.rmtree(path)
    #except OSError as x:
     #  print("Error occured: %s : %s" % (path, x.strerror))

    #os.mkdir(path)
    
else:
    def create_ceps_file(wavfile):
        sr, speech = scipy.io.wavfile.read(wavfile)
        #(sr)
        ceps=mfcc(speech)
        #ceps, mspec, spec= mfcc(song_array)
        #print(ceps.shape)

        bad = np.where(np.isnan(ceps))
        bad2=np.where(np.isinf(ceps))
        ceps[bad]=0
        ceps[bad2]=0
        write_ceps_file(ceps, wavfile)
    def write_ceps_file(ceps, wavfile):
        base_wav, ext = os.path.splitext(wavfile)
        data_wav = "file_tests/chunks/" + base_wav.split("/")[2] + ".ceps"
        np.save(data_wav, ceps)

    def read_ceps():
        X= []
        directory = "file_tests/chunks/"
        for filename in os.listdir(directory):
            #print(filename)
            if filename.endswith("npy"):
                ceps = np.load(directory + filename)
                num_ceps = len(ceps)
                X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))

        #print(np.array(X).shape)
        return np.array(X)

    path = sys.argv[1]
    #print(path)
    filename = path.split("/")[1]
    myaudio = AudioSegment.from_file(path , "m4a") 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    for i, chunk in enumerate(chunks):
        chunk_name = "file_tests/chunks/"+ filename +"{0}.wav".format(i)
        #print ("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")
    directory = "file_tests/chunks/"
    ceps_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and filename.endswith("wav"):
            #print(f)
            
            create_ceps_file(f)
    ceps_list = read_ceps()

    prediction = pickled_model.predict(ceps_list)
    #print(prediction)
    predict_percent = sum(prediction)/len(prediction)
    print(prediction)
    if predict_percent >= 0.5:
        print("The model predicts that this is a human speaking! There is approximately a " + str(round((predict_percent*100)))+"% chance of being real speech.")

    else:
        print("Warning, the model predicts that this is an AI voice!")
        print("There is approximately a " + str(100-round((predict_percent*100)))+"% chance of being AI-generated speech.")

import shutil
path = 'file_tests/chunks/'
try:
   shutil.rmtree(path)
except OSError as x:
   print("Error occured: %s : %s" % (path, x.strerror))

os.mkdir(path)
