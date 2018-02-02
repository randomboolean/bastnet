import os
import subprocess

tasks = []
graph = 'haxby_geo_6closest_symmetrized'
for order in [5,10]:
  for subject in [0,1,2,3,4,5]:
    tasks += [' "cd /homes/jcvialat/projects/bastnet;'
              + ' nohup '
              + ' python3 chebnet_haxby.py --subject {} --order {} --graph {}'.format(subject, order, graph)
              + ' > subject{}_K{}_{}.tmp &"'.format(subject, order, graph)]

servers = ['10.29.208.7' + str(i) for i in range(3, 10)]
servers += ['10.29.208.5' + str(i) for i in range(2, 10)]

while len(tasks) > 0:
  task = tasks.pop()

  foundAvailable = False
  while not(foundAvailable):
    if len(servers) == 0:
      print('No more available servers')
      print('Remaining tasks:')
      for t in tasks:
        print(t)
      sys.exit()
    server = servers.pop()
    queryGpu = ' "nvidia-smi --query-gpu=memory.free --format=csv,noheader"'
    proc = subprocess.Popen('ssh jcvialat@' + server + queryGpu, stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    mem = int(out[:-5])
    if mem > 6000:
      foundAvailable = True
  
  os.system('ssh jcvialat@' + server + task)
  print('Executed {} on {}'.format(task, server))

