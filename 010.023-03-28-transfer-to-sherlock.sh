#dang arg lists being too long to scp regularly...

for f in /sdf/scratch/kipac/delon/I_auto/comb_Lya_zmin_5.2_zmax_8.0*; do 
  echo $f
  cp $f /sdf/scratch/kipac/delon/SPHEREx
done

scp -r /sdf/scratch/kipac/delon/SPHEREx delon@sherlock:/scratch/users/delon/LIMxCMBL/I_auto/from_s3df/
