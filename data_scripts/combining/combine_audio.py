from scipy.io import wavfile
import os
import numpy as np

INPUT_NOISE_DIR = '../../data/raw_noise/'
INPUT_CLEAN_DIR = '../../data/sliced_clean/'
OUTPUT_DIR = '../../data/combined/'

for clean in os.listdir(INPUT_CLEAN_DIR)[:10]:
    if clean[-4:] == '.wav':
        for noise in os.listdir(INPUT_NOISE_DIR):
            if noise[-4:] == '.wav':
                rate_clean, data_clean = wavfile.read(INPUT_CLEAN_DIR + clean)
                rate_noise, data_noise = wavfile.read(INPUT_NOISE_DIR + noise)
                # import pdb; pdb.set_trace()
                # length = len(data_clean)

                # data_noise = data_noise[:length]
                m = min(len(data_clean), len(data_noise))
                average = np.average(np.array([data_clean[:m], data_noise[:m]]), axis=0)
                filename = OUTPUT_DIR + clean[:-4] + noise
                # average = [(s1/2 + s2/2) for (s1, s2) in zip(data_clean, data_noise)]
                # import pdb; pdb.set_trace()
                wavfile.write(filename, rate_clean, np.asarray(average, dtype=np.int16))

        print("wrote clean!", clean)
