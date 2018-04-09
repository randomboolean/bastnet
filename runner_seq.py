import os
import subprocess
import sys

tasks = []

if len(sys.argv) == 1:
  for order in [5,10,20]:
    for subject in [0,1,2,3,4,5]:
      graph = 'haxby_subject{}_kalofoliasprojected_4closest'.format(subject)
      tasks += ['/home/brain/miniconda3/bin/python -u chebnet_haxby_85.py --subject {} --order {} --graph {}'.format(subject, order, graph)]
else:
  tasks = open(sys.argv[1], 'r').readlines()

while len(tasks) > 0:
  
  #/home/brain/miniconda3/bin/python
  task = tasks.pop()
  os.system(task)
  print('Executed {}'.format(task))

