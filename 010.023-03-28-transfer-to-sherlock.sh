#dang arg lists being too long to scp regularly...

for f in /sdf/scratch/kipac/delon/I_auto/comb_HI_zmin_1.0_zmax_1.3*; do 
  echo $f
  cp $f /sdf/scratch/users/d/delon/CHIME/
done

scp -r /sdf/scratch/users/d/delon/CHIME delon@sherlock:/scratch/users/delon/LIMxCMBL/I_auto/from_s3df/
