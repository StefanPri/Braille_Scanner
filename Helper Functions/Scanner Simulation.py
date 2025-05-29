import os
import shutil
from PIL import Image

# construct sentence by inserting wanted sentence from database images
input_sentence = "Hi my name is Stefan."

flag = False
counter = 0

# first delete all entries
for file in os.listdir('Sentence Simulation/'):
    try:
        os.remove('Sentence Simulation/'+file)
    except :
        print(file)


for i in input_sentence:
    i = i.lower()

    # use for created data set 
    for file in os.listdir('Alphabet/'):
        letter = file[0]

        if letter == i and flag ==False:
            try:
                shutil.copy('Alphabet/'+file, 'Sentence Simulation/' + file)
                os.rename('Sentence Simulation/'+file,'Sentence Simulation/'+ str(counter)+'.jpg')
                counter +=1
                break
            except :
                print(file)
            flag = True
    flag = False




    



