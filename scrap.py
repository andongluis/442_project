import urllib
import urllib.request
import re
import random
import string
import csv
import numpy as np
import pandas as pd

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

#csv
a = np.chararray(shape = (5000, 5), itemsize=900, unicode = True)


fp = urllib.request.urlopen("https://www.instagram.com/explore/tags/lakemichigan/")
mybytes = fp.read()
mystr = mybytes.decode("utf8")
begin = mystr.find('window._sharedData = ')
end = mystr.find('}</script>')
mystr = mystr[0:end]
mystr = mystr[begin:len(mystr)]

counter = 0
srcs = [m.start() for m in re.finditer('https', mystr)]
links = []
for index in srcs:
    counter += 1
    copy_mystr = mystr[index:len(mystr)]
    likes = copy_mystr.find("count")
    for_likes = copy_mystr[likes:copy_mystr.find('}')]
    for_likes = for_likes[for_likes.find(':') :len(for_likes)]
    com = copy_mystr.find('com"')
    copy_mystr = copy_mystr[0:com]
    copy_mystr = copy_mystr + "com"
    name = for_likes + str('_') + randomString() + ".jpg"
    try:
        urllib.request.urlretrieve(copy_mystr, name)
    except:
        print("failed")
    print(str(counter) + copy_mystr)
    a[counter + 1][1] = copy_mystr
    a[counter + 1][2] = for_likes

print(srcs)
df = pd.DataFrame(a)
df.to_csv("all_data.csv")


urllib.request.urlretrieve("https://scontent-ort2-2.cdninstagram.com/vp/229b46e6f9e62ee8587afad97edc61f2/5D1CEC1F/t51.2885-15/e35/s480x480/29740677_131742461010120_5463307634415239168_n.jpg?_nc_ht=scontent-ort2-2.cdninstagram.com", "local-filename.jpg")
