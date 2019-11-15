import pywt
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from os import listdir
from numba import cuda

class Wavelet_deblur:

    def __init__(self, filepath):
        self.filepath = filepath
        self.pictures_list = listdir(filepath)

    def show_swt(self, data_or_name_or_index, l=None):
        # data_or_name_or_index can be wavelet coefficients, name of the file or index of file in folder
        # l is decomposition level
        # plots the coefficients of level l

        if type(data_or_name_or_index) is list:
            coefs = data_or_name_or_index
        else:
            name_or_index = data_or_name_or_index
            coefs = self.get_swt_coeffs(name_or_index)
        N = len(coefs) - 1
        LL, (LH, HL, HH) = coefs[0], coefs[N-l]
        titles = ['LL max level ({})'.format(N-1), 'LH level {}'.format(l), 'HL level {}'.format(l), 'HH level {}'.format(l)]
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest",cmap='gray')
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
        plt.show()

    # picture name or index
    def get_swt_coeffs(self, name_or_index,wavelet = 'db1'):
        if type(name_or_index) is int:
            index = name_or_index
            p = cv2.imread(self.filepath + self.pictures_list[index], cv2.IMREAD_GRAYSCALE)
        else:
            name = name_or_index
            p = cv2.imread(self.filepath + name, cv2.IMREAD_GRAYSCALE)
        p = pywt.dwt2(p,'db1')[0]
        l_max = pywt.swt_max_level(min(np.shape(p)))
        self.max_level = l_max

        # Multilevel 2D stationary wavelet transform with daubechies filters.
        coefs = pywt.swt2(p, wavelet, l_max, start_level=0, trim_approx=True, norm= True)

        # converting from float64 to uint8 to save memory
        coefs[0] = coefs[0].astype(np.int)
        for level in range(1, l_max):
            coefs[level] = list(coefs[level])
            for band in range(3):
                picture = np.asarray(coefs[level][band])
                picture -= np.min(picture)
                picture *= 255/np.max(picture)
                picture = picture.astype(np.uint8)
                coefs[level][band] = picture
        return coefs

    def image_float2int(self,img):
        print("")

    def MAD(self, b1_x, b1_y, l_prev, l_cur, N):
        n_b_x = int(np.shape(l_cur)[0]/N)
        n_b_y = int(np.shape(l_cur)[1]/N)
        s = []
        block1 = l_prev[b1_x * N:(b1_x + 1) * N, b1_y * N:(b1_y + 1) * N]
        for b2_x in range(n_b_x):
            for b2_y in range(n_b_y):
                block2 = l_cur[b2_x*N:(b2_x+1)*N,b2_y*N:(b2_y+1)*N]
                s.append(int(np.sum(np.abs(np.add(block1,block2*-1))) / (N*N)))
        return np.argmin(s)

    def MAD2(self, b1_x, b1_y, l_prev, l_cur, N):
        n_b_x = int(np.shape(l_cur)[0]/N)
        n_b_y = int(np.shape(l_cur)[1]/N)
        n_b = n_b_x * n_b_y
        s = []
        block1 = l_prev[b1_x * N:(b1_x + 1) * N, b1_y * N:(b1_y + 1) * N]
        for i in range(n_b):
            block2 = l_cur[(i%n_b_x)*N:(i%n_b_x+1)*N,(int(i/n_b_y))*N:(int(i/n_b_y)+1)*N]
            s.append(np.sum([abs(block1[j%N, int(j/N)] - block2[j%N, int(j/N)]) for j in range(N*N)])/(N*N))
        return np.argmin(s)

    def MAD3(self, b1_x, b1_y, b2_x, b2_y, l_prev, l_cur, N):
        block1 = l_prev[b1_x * N:(b1_x + 1) * N, b1_y * N:(b1_y + 1) * N]
        block2 = l_cur[b2_x * N:(b2_x + 1) * N, b2_y * N:(b2_y + 1) * N]
        s = np.sum([abs(block1[j % N, int(j / N)] - block2[j % N, int(j / N)]) for j in range(N * N)]) / (N * N)
        return np.argmin(s)

    def three_step_search(self, l_prev, l_cur, N):
        # volgens "THREE STEP SEARCH METHOD FOR BLOCK MATCHING ALGORITHM"
        # url: http://www.digitalxplore.org/up_proc/pdf/62-1397565973101-104.pdf

        if N / 2 == int(N / 2):
            print("N must be odd")
            return

        n_steps = int(np.log2(N+1))
        w_steps = [pow(2, n_steps - (i+1)) for i in range(n_steps)]
        n_b_x = int(np.shape(l_cur)[0] / N)
        n_b_y = int(np.shape(l_cur)[1] / N)
        n_b = n_b_x * n_b_y
        N_2 = int(N / 2)

        MV = []
        n_b = 100 # om te testen
        for i in range(n_b):
            block1 = l_prev[(i % n_b_x) * N:((i + 1) % n_b_x) * N, int(i / n_b_y) * N:(int(i / n_b_y) + 1) * N]

            #1e centrum is centrum van block1
            c_x, c_y = i % n_b_x + N_2, int((i / n_b_y) + N_2)
            oc_x,oc_y = c_x, c_y
            mad_glob = []
            for w in w_steps:
                mad_loc = []
                # 8 punten rond centrum
                centers = [(c_x - w, c_y - w), (c_x - w, c_y),(c_x - w, c_y + w),(c_x, c_y - w),
                           (c_x, c_y + w), (c_x + w, c_y - w), (c_x + w, c_y), (c_x + w, c_y + w)]
                for c in centers:
                    block2 = l_cur[c[0] - N_2: c[0] + N_2 + 1, c[1] - N_2: c[1] + N_2 + 1]
                    # MAD-functie op block 1 en 2
                    # toevoegen aan mad_loc (verzameling mad's binnen huidige iteratie)
                    # kan fout gaan als block 2 buiten de afbeelding zit. in dat geval gewoon niets doen
                    try:
                        mad_loc.append({"mad": np.sum([abs(int(block1[j%N, int(j/N)]) - int(block2[j%N, int(j/N)]))/(N*N) for j in range(N*N)]), "c": c, "oc": (oc_x,oc_y)})
                    except:
                        pass
                # soms kunnen ze allemaal mislukken aan de randen als het venster te groot is
                try:
                    # centrum met kleinste mad is nieuwe centrum voor volgende iteratie
                    mad_loc_argmin = np.argmin(list(map(lambda x: x["mad"], mad_loc)))
                    c_x, c_y = mad_loc[mad_loc_argmin]["c"]
                    # kleinste mad toevoegen aan mad_glob
                    mad_glob.append(mad_loc[mad_loc_argmin])
                except:
                    pass
            # kleinste mad van alle lokale minima teruggeven
            try:
                MV.append(mad_glob[np.argmin(list(map(lambda x: x["mad"], mad_glob)))])
            except:
                pass
            progress = 100*i/n_b
            if progress % 10 == 0:
                print("\rprogress: {}%".format(progress), end='')
        return MV

filepath = "../testbeelden/"
wd = Wavelet_deblur(filepath)
coefs_prev = wd.get_swt_coeffs(0)
coefs_cur = wd.get_swt_coeffs(1)

# print("start")
# start_time = time.time()
# vms_pot = wd.MAD(0, 0, coefs_prev[1][1], coefs_cur[1][1], 10)
# print(vms_pot)
# print("--- %s seconds ---" % (time.time() - start_time))

# sum = 0
# for i in range(150*200-1):
#     sum += 150*200-i
# print(0.32*sum/(3600*30000))


l = 1
b = 0
start_time = time.time()

vectors = wd.three_step_search(coefs_prev[l][b], coefs_cur[l][b], 31)
print("--- %s seconds ---" % (time.time() - start_time))
print(vectors)

plt.subplot(121)
plt.imshow(coefs_prev[l][b], interpolation="nearest",cmap='gray')
for v in vectors:
    plt.plot([v["oc"][0], v["c"][0]],[v["oc"][1], v["c"][1]])
plt.subplot(122)
plt.imshow(coefs_cur[l][b], interpolation="nearest",cmap='gray')
for v in vectors:
    plt.plot([v["oc"][0], v["c"][0]],[v["oc"][1], v["c"][1]])
plt.show()
