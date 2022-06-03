import MDAnalysis as mda
import prolif as plf
import multiprocessing as mp
from tqdm.auto import tqdm
import numpy as np

# load trajectory
u = mda.Universe(plf.datafiles.TOP, plf.datafiles.TRAJ)
# create selections for the ligand and protein
lig = u.select_atoms("resname LIG")
prot = u.select_atoms("protein and byres around 7.0 group lig", lig=lig, updating=True)

# # serial
# fp = plf.Fingerprint()
# fp.run(u.trajectory[[0, 10, 20]], lig, prot)
# df = fp.to_dataframe()
# print(df)

# parallel
N_WORKERS = 6
frames = range(u.trajectory.n_frames)
chunks = np.array_split(frames, N_WORKERS)

def job(chunk):
    univ = u.copy()
    lig = univ.select_atoms("resname LIG")
    prot = univ.select_atoms("protein")
    fp = plf.Fingerprint()
    fp.run(univ.trajectory[chunk], lig, prot, progress=chunk[0]==0)
    return fp.ifp

with mp.Pool(N_WORKERS) as pool:
    results = []
    for ifp in tqdm(pool.imap_unordered(job, chunks),
                    total=N_WORKERS):
        results.extend(ifp)
    
df = plf.to_dataframe(results, plf.Fingerprint().interactions.keys())
df.sort_index(inplace=True)
print(df)
