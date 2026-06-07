import torch
import json
def merge(path1, path2, path3):
    p1=torch.load(path1)
    p2=torch.load(path2)
    p3=torch.cat([p1,p2])
    torch.save(p3,path3)

def merge_json(path1,path2,path3):
    with open(path1, "r") as f:
        j1=json.load(f)
    with open(path2, "r") as f:
        j2=json.load(f)
    added = max([int(i) for i in list(j1.keys())])+1
    j3 = {str(int(k)+added): v for k,v in j2.items()}
    j4 = j1 | j3
    with open(path3, "w") as f:
        json.dump(j4,f)