import os
from tqdm import tqdm

folders = [
    "data/cvrp/augerat/a-vrp",
    "data/cvrp/augerat/b-vrp",
    "data/cvrp/fisher"]

for folder in folders:
    for files in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, files)
        for i in range(5):
            os.system(
                f"python src/main.py cvrp {file_path} " +
                "--output results/results.csv -g 10000 -p 3000 -mp True")
    print('finished folder %s' % folder)
