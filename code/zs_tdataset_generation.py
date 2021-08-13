import random

knownR = "testpoolSI.txt" #file of seen relations pool
testfile = "TESTFILE.txt" #Your own test file
unknownR = "zeroshotSI.txt" #file of unseen relations pool
n_known = 4000 #number of seen relations
n_unknown = 1000 #number of unseen relations

with open(unknownR, "r") as f:
    lines_unknown = f.readlines()

with open(knownR, "r") as fk:
    lines_known = fk.readlines()


with open(testfile, "a+") as f1:
    for _ in range(n_known):
        f1.write(lines_known.pop(random.randint(0, len(lines_known) - 1)))


with open(testfile, "a+") as f2:
    for _ in range(n_unknown):
        f2.write(lines_unknown.pop(random.randint(0, len(lines_unknown) - 1)))
