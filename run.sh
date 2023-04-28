source /cluster/apps/local/env2lmod.sh && module load gcc/6.3.0 python_gpu/3.8.5
source /cluster/project/cvl/admin/cvl_settings
source /cluster/home/leisun/base/bin/activate

cd /cluster/home/${USER}/EventDeblurProject/Davis346_inference
echo "copying codes"
rsync -aq ./ ${TMPDIR}
cd $TMPDIR

python inference.py
# python npz_to_h5.py
