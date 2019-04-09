import os
from tinydb import TinyDB, Query

for root, dirs, files in os.walk('trainingImages', topdown = False):
	i = 0	
	list = {}
	for name in dirs:
		print os.path.join(name)
		list[i] = os.path.join(name)
		i = i+1
print list

db = TinyDB('users.json')
db.insert(list)
