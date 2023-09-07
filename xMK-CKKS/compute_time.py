import os

def time(name,data):
    with open(name,"a+") as f:
        f.write(data)