import sys


idx = int(sys.argv[1])
nbins = int(sys.argv[2])


_tmp = 0
found = False

for idx1 in range(nbins):
    for idx2 in range(idx1, nbins):
        if(_tmp == idx):
            print(idx1, idx2)
            found = True
            break
        _tmp += 1
    if(found):
        break

