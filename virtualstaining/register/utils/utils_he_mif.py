import os
import time
from valis import registration

def excute_register(he_path, mif_path, src_dir, results_dst_dir, registered_slide_dst_dir):
    start = time.time()
    registrar = registration.Valis(
        src_dir=src_dir,
        dst_dir=results_dst_dir,
        img_list=[he_path, mif_path],
        reference_img_f=mif_path,
        align_to_reference=True
    )
    registrar.register()
    registrar.warp_and_save_slides(registered_slide_dst_dir, crop="overlap")


