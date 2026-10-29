# Analysis structures
`topology_apo16e6.pdb` is the 151-residue analysis topology shared by the unbiased trajectories, with CYS at positions 80, 97, 111 and 140. It is not represented as a fully parameterized initial Desmond system. The original prepared input CMS must be supplied through Zenodo because of its size.

`4XR8_clean.pdb` and `7UAJ_clean.pdb` are the reference inputs actually used in Table S1. `hpacker-openmm_sidechain_rec.pdb` and `clustered_topology.pdb` are the full-side-chain and reduced BioEmu topology inputs. Preserve their atom orders when obtaining matching XTC files. Reference structures: https://www.rcsb.org/structure/4XR8 and https://www.rcsb.org/structure/7UAJ .
