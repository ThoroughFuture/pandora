import os
import glob
from utils.utils_he_mif import excute_register
from valis import registration
import valis
import time

root = "path/to/he_mif_root"
registered_slide_dst_dir = "path/to//mif_registered_data_mifref"
results_dst_dir = "path/to/mif_registered_data_view_mifref"
os.makedirs(results_dst_dir, exist_ok=True)
os.makedirs(registered_slide_dst_dir, exist_ok=True)

pair_list = os.listdir(root)
pair_list.sort()

pair_list = ['1', '5', '14', '18']
print(pair_list, flush=True)

for idx in pair_list:
    src_dir = f"{root}/{idx}"
    mif_pth = glob.glob(f"{src_dir}/*CODEX*.qptiff")[0]

    he_pth = glob.glob(f"{src_dir}/*HE*.qptiff")
    if not he_pth:
        he_pth = glob.glob(f"{src_dir}/*.vsi")
    he_pth = he_pth[0]
    print(mif_pth, flush=True)
    print(he_pth, flush=True)

    try:
        excute_register(he_path=he_pth, mif_path=mif_pth, src_dir=src_dir, results_dst_dir=results_dst_dir,
                        registered_slide_dst_dir=registered_slide_dst_dir)
    except Exception as e:
        print('process_error', e, flush=True)
        print(flush=True)
        continue

    print(f"{idx}th Done")
    print(flush=True)

# Kill the JVM
registration.kill_jvm()
