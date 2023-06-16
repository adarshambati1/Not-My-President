from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from pathlib import Path


directory = 'audioclips/real_original'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and filename.endswith("m4a"):
        print(f)
        myaudio = AudioSegment.from_file(f , "m4a") 
        chunk_length_ms = 10000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

        for i, chunk in enumerate(chunks):
            chunk_name = "audioclips/real_clean/"+ filename +"{0}.wav".format(i)
            print ("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")