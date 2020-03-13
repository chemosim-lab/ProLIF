with open("/data/ProLIF/all.pdb","w") as f:
    for frame in lig_traj:
        i = frame.n_frame + 1
        f.write(f"MODEL{i:9d}\n")
        mol = Chem.MolToPDBBlock(frame).split("\n")[:-2]
        mol = [l for l in mol if not l.startswith("CONECT")]
        mol = "\n".join(mol)
        f.write("%s\n"%mol)
        f.write("ENDMDL\n")
    conect = Chem.MolToPDBBlock(frame).split("\n")[:-2]
    conect = [l for l in conect if l.startswith("CONECT")]
    conect = "\n".join(conect)
    f.write(conect+"\n")
    f.write("END")
