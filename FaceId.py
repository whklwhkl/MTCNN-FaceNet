import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as patches
from PIL import Image
import detection
import recognition
import os
import argparse

# build dataset
print("constructing the dataset")
lf = [o for o in os.listdir("TrainImage")]
imgnet = {k:np.array(Image.open(os.path.join("TrainImage",k))) for k in lf if 'jpg' in k}
d_mean = np.vstack(imgnet.values()).mean(axis=0)
dataset = {k:recognition.get_vec(imgnet[k]-d_mean) for k in lf if 'jpg' in k}
print("construction complete")

def main(f_image="testing/IMG_1819.jpg",output="RESULT.png",K=3,columns=5,width=1152,height=864):
    # f_image='testing/IMG_1819.jpg'
    #testing/png/IMG_1818.png
    # testing/IMG_1818.jpg
    tic = time.time()

    i= Image.open(f_image)
    i=i.resize((width,height))
    i=i.convert("RGB")
    bindBoxes=detection.get_bindingBoxes(i)

    # recognition
    vecs = np.vstack([recognition.get_vec(
        np.array(
            i.crop(bb).resize(
                (160,160)
            )
        )-d_mean) for bb in bindBoxes[:,:-1]])

    labs = np.array(list(dataset.keys()))
    # K=3
    func = sp.spatial.distance.cosine#euclidean
    whos = []
    for v in vecs:
        dd = np.vstack([func(v,dataset[k])for k in labs]) # distance
        persons = np.argpartition(dd,K,axis=0)[:K]
        whos.append(persons)
    def find_the_most_frequent(list_of_string):
        c={}
        set_of_string = set(list_of_string)
        if len(set_of_string)==len(list_of_string):return None
        for e in set_of_string:
            c[e]=0
        for e in list_of_string:
            c[e]+=1
        return max(c,key=c.get)
    rrr = []
    for j,w in enumerate(whos):
        names_people = []
        for h in labs[w]:names_people.append(h[0].split('_')[0])
        result = find_the_most_frequent(names_people)
        rrr.append(result if result else names_people[0])
    # ### Generate LaTex code for Recognition Result
    # columns = 5
    print('''
\\begin{table}[]
\\centering
\\caption{Result of recognition, where UID in \\textbf{bold} is successfully identified in the photo}
\\label{tbl:rrfnknn}''')
    print('\\begin{tabular}{%s}'%('|'.join(["ll"]*columns)))
    print('%s \\\\'%(' & '.join(["Index &   UID"]*columns)),'\\hline')
    while len(rrr)%columns!=0:
        rrr.append(' ')
    for j in range(len(rrr)//columns):
        for k in range(columns):
            ind = j*columns+k
            print((""if not k else "& ")+"{:<2} & {:<8} ".format(ind if rrr[ind] is not ' 'else ' ',rrr[ind]),end='')
        print("\\\\")
    print('''\\end{tabular}
\\end{table}
\\clearpage''')

    def cast_4p_to_tlhw(p):
        '''helper function organizing the order of bounding boxes\' parameters '''
        return p[1],p[0],p[3]-p[1],p[2]-p[0]

    fig, ax = plt.subplots()

    def draw_binding_box(t,l,h,w):
        rect = patches.Rectangle((l,t),w,h,edgecolor='g',facecolor='none')
        ax.add_patch(rect)

    ax.set_axis_off()
    fig.set_size_inches((20,15))
    ax.imshow(i,extent=None)
    for count,p5 in enumerate(bindBoxes):
        t,l,h,w=cast_4p_to_tlhw(p5)
        draw_binding_box(t,l,h,w)
        plt.text(l,t,str(count),color='c')
    plt.savefig(output)

    elapse = int(time.time()-tic)
    print("\nthe running time is %d seconds"%(elapse%60))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configurate the system')
    parser.add_argument('-i', '--image',
        help='the [i]mage used for testing',
        default='testing/IMG_1819.jpg')
    parser.add_argument('-k', '--knn',
        help='the parameter [k] used in KNN when identifying the person',
        default=3)
    parser.add_argument('-c', '--columns',
        help='the number of [c]olumns defined when drawing the result table',
        default=5)
    parser.add_argument('-o', '--output',
        help='the path indicating where to save the result image',
        default="RESULT.png")
    args = parser.parse_args()
    main(args.image,
        args.output,
        args.knn,
        args.columns)
