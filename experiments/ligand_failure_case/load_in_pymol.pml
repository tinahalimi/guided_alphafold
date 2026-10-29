# open from the experiment folder:  pymol load_in_pymol.pml
set normalize_ccp4_maps, 0
load gt/8a2b_gt_model.pdb, gt_holo
load gt/4i20_apo_in_8a2b_frame.pdb, apo_4i20
load gt/8a2b_gt_density.ccp4, gt_map
isomesh gt_mesh, gt_map, 1.0
color grey70, gt_mesh
load outputs/final/guided_sample0.pdb, guided_s0
load outputs/final/guided_sample1.pdb, guided_s1
load outputs/final/guided_sample2.pdb, guided_s2
load outputs/final/guided_sample3.pdb, guided_s3
load outputs/final/guided_sample4.pdb, guided_s4
load outputs/final/guided_sample0.ccp4, guided_s0_map
isomesh guided_s0_mesh, guided_s0_map, 1.0
color skyblue, guided_s0_mesh
disable guided_s0_mesh
color skyblue, guided_s* and elem C
load outputs/final/unguided_sample0.pdb, unguided_s0
load outputs/final/unguided_sample1.pdb, unguided_s1
load outputs/final/unguided_sample2.pdb, unguided_s2
load outputs/final/unguided_sample3.pdb, unguided_s3
load outputs/final/unguided_sample4.pdb, unguided_s4
load outputs/final/unguided_sample0.ccp4, unguided_s0_map
isomesh unguided_s0_mesh, unguided_s0_map, 1.0
color wheat, unguided_s0_mesh
disable unguided_s0_mesh
color wheat, unguided_s* and elem C
hide everything
show cartoon
show sticks, resn KY9
show lines, byres (resn KY9 around 5)
color magenta, gt_holo and resn KY9 and elem C
orient resn KY9
# checkpoints: outputs/checkpoints/step*/ (step199 = final)
