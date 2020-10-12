import os


files = [f for f in os.listdir('.') if os.path.isfile(f)]

for f in files:
    if 'subject' in f and '.bmp' not in f:
        os.rename(f, f+'.bmp')