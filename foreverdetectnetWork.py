#!/usr/bin/python
from subprocess import Popen
import sys

filename = "detectnetWork.py"

while True:
    print("\nStarting " + filename)
    p = Popen("sudo python3 " + filename, shell=True)
    aa=p.wait()
