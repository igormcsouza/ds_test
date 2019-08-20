from PIL import Image

# Reading the inputs!
def reading_file_name(base_dir):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
    return onlyfiles

test_files_name = reading_file_name('Large Files/test.rotfaces/test/')

from pandas import read_csv
labels = read_csv('Large Files/test.rotfaces/test/test.preds.csv')

Ya = []
for item in labels.label:
    Ya.append(item)

from os.path import join, isdir
from os import getcwd, makedirs

for item in test_files_name:
    image  = Image.open(join('Large Files/test.rotfaces/test/' + item))

save_dir = join(getcwd(), 'saved_models')

if not isdir(save_dir):
    makedirs(save_dir)