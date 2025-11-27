#dang arg lists being too long to scp regularly...

for f in /sdf/scratch/kipac/delon/I_auto/comb_HI_zmin_0.8_zmax_2.5_*; do 
  echo $f
  cp $f /sdf/scratch/kipac/delon/CHIME
done

scp -r /sdf/scratch/kipac/delon/CHIME delon@sherlock:/scratch/users/delon/LIMxCMBL/I_auto/from_s3df/
