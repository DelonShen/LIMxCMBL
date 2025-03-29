#dang arg lists being too long to scp regularly...

for f in /sdf/scratch/kipac/delon/I_auto/comb_CO_zmin_2.4_zmax_3.4*; do 
  echo $f
  cp $f /sdf/scratch/users/d/delon/COMAP/
done

scp -r /sdf/scratch/users/d/delon/COMAP delon@sherlock:/scratch/users/delon/LIMxCMBL/I_auto/from_s3df/
