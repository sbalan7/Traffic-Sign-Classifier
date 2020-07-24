import matplotlib.pyplot as plt
import numpy as np
import os


def count_classes():
    train_dir_path = 'C:\\Users\\sbala\\Google Drive\\Datasets\\Traffic_Signs\\Train'
    train_dir = os.listdir(train_dir_path)
    no_obj = []

    for path in train_dir:
        curr = os.path.join(train_dir_path, path)
        no_obj.append(len(os.listdir(curr)))
    
    fig, ax = plt.subplots()
    ax.bar(np.arange(0, len(no_obj)), no_obj, color='#ce4873')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Number of images per class')
    ax.set_xticks(np.arange(0, len(classes)))
    ax.set_xticklabels(classes.values())
    ax.yaxis.grid('on')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        tick.label.set_rotation('vertical')
    plt.show()

def plot_sample(rows, cols):
    train_dir_path = 'C:\\Users\\sbala\\Google Drive\\Datasets\\Traffic_Signs\\Train'
    train_dir = os.listdir(train_dir_path)
    selection = np.random.randint(0, 43, rows*cols)
    
    fig = plt.subplots(rows, cols, figsize=(cols+1, rows+1))
    i = 1

    for sel in selection:
        curr = os.path.join(train_dir_path, str(sel))
        image = plt.imread(os.path.join(curr, np.random.choice(os.listdir(curr))))
        plt.subplot(rows, cols, i)
        i += 1
        plt.imshow(image)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(classes[sel+1], fontsize=6)
    plt.show()


classes = { 
    1:'Speed limit (20km/h)',
    2:'Speed limit (30km/h)', 
    3:'Speed limit (50km/h)', 
    4:'Speed limit (60km/h)', 
    5:'Speed limit (70km/h)', 
    6:'Speed limit (80km/h)', 
    7:'End of speed limit (80km/h)', 
    8:'Speed limit (100km/h)', 
    9:'Speed limit (120km/h)', 
    10:'No passing', 
    11:'No passing vehicles over 3.5 tons', 
    12:'Right-of-way at intersection', 
    13:'Priority road', 
    14:'Yield', 
    15:'Stop', 
    16:'No vehicles', 
    17:'Vehicles > 3.5 tons prohibited', 
    18:'No entry', 
    19:'General caution', 
    20:'Dangerous curve left', 
    21:'Dangerous curve right', 
    22:'Double curve', 
    23:'Bumpy road', 
    24:'Slippery road', 
    25:'Road narrows on the right', 
    26:'Road work', 
    27:'Traffic signals', 
    28:'Pedestrians', 
    29:'Children crossing', 
    30:'Bicycles crossing', 
    31:'Beware of ice/snow',
    32:'Wild animals crossing', 
    33:'End speed + passing limits', 
    34:'Turn right ahead', 
    35:'Turn left ahead', 
    36:'Ahead only', 
    37:'Go straight or right', 
    38:'Go straight or left', 
    39:'Keep right', 
    40:'Keep left', 
    41:'Roundabout mandatory', 
    42:'End of no passing', 
    43:'End no passing vehicle > 3.5 tons' 
}


