import os
import subprocess
import sys

tasks = []
order0, subject0 = (10, 3)

if len(sys.argv) == 1:
  graph = 'haxby_geo_6closest_symmetrized'
  graph = 'haxbySubject0_kalofoliasProjected_4closest_symmetrized'
  for order in [5,10,15]:
    if order0 < order:
      break
    for subject in [0,1,2,3,4,5]:
      if order0 == order and subject0 == subject:
        break
      tasks += [' "cd jc/bastnet;'
                + ' nohup '
                + ' /home/brain/miniconda3/bin/python chebnet_haxby.py --subject {} --order {} --graph {}'.format(subject, order, graph)
                + ' > subject{}_K{}_{}.tmp &"'.format(subject, order, graph)]
else:
  tasks = open(sys.argv[1], 'r').readlines()

# servers = ['10.29.208.7' + str(i) for i in range(3, 10)]
# servers += ['10.29.208.5' + str(i) for i in range(2, 10)]

servers = ['10.29.232.' + str(i) for i in [51, 68, 90, 138, 141, 199, 235, 245, 248]]

#os.system('rm -rf checkpoints/*')
#os.system('rm -rf summaries/*')

while len(tasks) > 0:
  task = tasks.pop()
  foundAvailable = False
  while not(foundAvailable):
    if len(servers) == 0:
      print('No more available servers')
      print('Remaining tasks:')
      print(task)
      for t in tasks:
        print(t)
      sys.exit()
    server = servers.pop()
    queryGpu = ' "nvidia-smi --query-gpu=memory.free --format=csv,noheader"'
    proc = subprocess.Popen('ssh brain@' + server + queryGpu, stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    if len(out) > 0:
      mem = int(out[:-5])
      if mem > 3000:
        foundAvailable = True
  
  #/home/brain/miniconda3/bin/python
  os.system('ssh brain@' + server + task)
  print('Executed {} on {}'.format(task, server))

