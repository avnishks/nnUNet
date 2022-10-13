# from ntpath import join
from fileinput import filename
from unicodedata import name
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import subprocess
from pathlib import Path

import SimpleITK as sitk
from nnunet.paths import nnUNet_raw_data
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.utilities.sitk_stuff import copy_geometry


def convert_labels_back_to_limbic(source_nifti, target_nifti):
    nnunet_itk = sitk.ReadImage(source_nifti)
    nnunet_npy = sitk.GetArrayFromImage(nnunet_itk)
    # limbic_seg = np.zeros(nnunet_npy.shape, dtype=np.uint8)
    # limbic_seg[nnunet_npy == 1] = 26   # Left-Nucleus-Accumbens
    # limbic_seg[nnunet_npy == 2] = 58   # Right-Nucleus-Accumbens
    # limbic_seg[nnunet_npy == 3] = 819  # Left-HypoThal-noM
    # limbic_seg[nnunet_npy == 4] = 820  # Right-HypoThal-noMB
    # limbic_seg[nnunet_npy == 5] = 821  # Left-Fornix
    # limbic_seg[nnunet_npy == 6] = 822  # Right-Fornix
    # limbic_seg[nnunet_npy == 7] = 843  # Left-MammillaryBody
    # limbic_seg[nnunet_npy == 8] = 844  # Right-MammillaryBody
    # limbic_seg[nnunet_npy == 9] = 853  # Mid-AntCom
    # limbic_seg[nnunet_npy == 10] = 865 # Left-Basal-Forebrain
    # limbic_seg[nnunet_npy == 11] = 866 # Right-Basal-Forebrain
    # limbic_seg[nnunet_npy == 12] = 869 # Left-SeptalNuc
    # limbic_seg[nnunet_npy == 13] = 870 # Right-SeptalNuc
    
    limbic_seg = np.zeros(nnunet_npy.shape, dtype=np.uint16)
    limbic_seg[nnunet_npy == 26] = 26   # Left-Nucleus-Accumbens
    limbic_seg[nnunet_npy == 58] = 58   # Right-Nucleus-Accumbens
    limbic_seg[nnunet_npy == 51] = 819  # Left-HypoThal-noM
    limbic_seg[nnunet_npy == 52] = 820  # Right-HypoThal-noMB
    limbic_seg[nnunet_npy == 53] = 821  # Left-Fornix
    limbic_seg[nnunet_npy == 54] = 822  # Right-Fornix
    limbic_seg[nnunet_npy == 75] = 843  # Left-MammillaryBody
    limbic_seg[nnunet_npy == 76] = 844  # Right-MammillaryBody
    limbic_seg[nnunet_npy == 85] = 853  # Mid-AntCom
    limbic_seg[nnunet_npy == 97] = 865 # Left-Basal-Forebrain
    limbic_seg[nnunet_npy == 98] = 866 # Right-Basal-Forebrain
    limbic_seg[nnunet_npy == 101] = 869 # Left-SeptalNuc
    limbic_seg[nnunet_npy == 102] = 870 # Right-SeptalNuc
    limbic_seg_itk = sitk.GetImageFromArray(limbic_seg)
    limbic_seg_itk = copy_geometry(limbic_seg_itk, nnunet_itk)
    sitk.WriteImage(limbic_seg_itk, target_nifti)

def convert_to_nnunet_labels(source_nifti, target_nifti):
    nnunet_itk = sitk.ReadImage(source_nifti)
    nnunet_npy = sitk.GetArrayFromImage(nnunet_itk)
    limbic_seg = np.zeros(nnunet_npy.shape, dtype=np.uint8)
    limbic_seg[nnunet_npy == 26] = 1   # Left-Nucleus-Accumbens
    limbic_seg[nnunet_npy == 58] = 2   # Right-Nucleus-Accumbens
    limbic_seg[nnunet_npy == 51] = 3  # Left-HypoThal-noM
    limbic_seg[nnunet_npy == 52] = 4  # Right-HypoThal-noMB
    limbic_seg[nnunet_npy == 53] = 5  # Left-Fornix
    limbic_seg[nnunet_npy == 54] = 6  # Right-Fornix
    limbic_seg[nnunet_npy == 75] = 7  # Left-MammillaryBody
    limbic_seg[nnunet_npy == 76] = 8  # Right-MammillaryBody
    limbic_seg[nnunet_npy == 85] = 9  # Mid-AntCom
    limbic_seg[nnunet_npy == 97] = 10 # Left-Basal-Forebrain
    limbic_seg[nnunet_npy == 98] = 11 # Right-Basal-Forebrain
    limbic_seg[nnunet_npy == 101] = 12 # Left-SeptalNuc
    limbic_seg[nnunet_npy == 102] = 13 # Right-SeptalNuc
    limbic_seg_itk = sitk.GetImageFromArray(limbic_seg)
    limbic_seg_itk = copy_geometry(limbic_seg_itk, nnunet_itk)
    sitk.WriteImage(limbic_seg_itk, target_nifti)

def rename_files_for_task(folder, task_name, *, renaming_key_path='', image_or_label='image'):
    if image_or_label == 'image':
        files = (Path(root + '/' + filename)
            for root, _, filenames in os.walk(folder)
            for filename in filenames    
        )

        for idx, file in enumerate(files):
            print("renaming: ", file.name)
            new_name = str(file.parent) + '/' + task_name + f'_{idx+1:03d}' + '_0000' + '.nii.gz'
            file.replace(new_name)
            renaming_key[file.name] = Path(new_name).name
    
    elif image_or_label == 'label':
        files = (Path(root + '/' + filename)
            for root, _, filenames in os.walk(folder)
            for filename in filenames    
        )

        if renaming_key_path=='':
            renaming_key={}
            for idx, file in enumerate(files):
                print("renaming: ", file.name)
                new_name = str(file.parent) + '/' + task_name + f'_{idx+1:03d}' + '.nii.gz'
                file.replace(new_name)
                renaming_key[file.name] = Path(new_name).name
        else:
            with open(renaming_key_path, 'r') as f:
                renaming_key = json.load(f)
            
            # reverse_naming_key = dict((v,k) for k,v in renaming_key.items())
            count =0
            for idx, file in enumerate(files):
                # print(f'file_idx:{idx} and name {file}')
                if renaming_key.__contains__(str(Path(file.name))):
                    rename_idx = renaming_key[str(Path(file.name))][1].split('_')[1].split('.')[0]
                    new_name = str(file.parent) + '/' + task_name + f'_{rename_idx}' + '.nii.gz'
                    # print("old filenames: ", str(Path(file.name)))
                    # print("new filename: ", new_name)
                    file.replace(new_name)
                    count += 1
            print(f"Renamed {count} files.")
    else:
        print ("Not implemented error!")

def rename_files_with_key(folder, renaming_dict_path):
    with open(renaming_dict_path, 'r') as f:
        renaming_dict = json.load(f)

    files = (Path(root + '/' + filename)
        for root, _, filenames in os.walk(folder)
        for filename in filenames    
    )
    len_files = sum(1 for _ in files)
    print("len(files): ", len_files)
    asdf = list(files)
    print(asdf)

    reverse_naming_dict = dict((v,k) for k,v in renaming_dict.items())
    count =0
    for idx, file in enumerate(files):
        print(idx, file)
        if reverse_naming_dict.__contains__(str(Path(Path(file.stem).stem).stem) + '.nii.gz'):
            print(renaming_dict[str(Path(file.stem).stem) + '.nii.gz'])
            # file.replace(renaming_dict[str(Path(file.stem).stem) + '.nii.gz'])
            count += 1
    print("Done renaming {} files".format(count+1))

def rename_files_back_to_native_names(folder, renaming_dict_path):
    with open(renaming_dict_path, 'r') as f:
        renaming_dict = json.load(f)
    
    files = (Path(root + '/' + filename)
        for root, _, filenames in os.walk(folder)
        for filename in filenames    
    )

    reverse_naming_dict = dict((v,k) for k,v in renaming_dict.items())
    count =0
    for idx, file in enumerate(files):
        if reverse_naming_dict.__contains__(str(Path(file.stem).stem) + '_0000.nii.gz'):
            file.replace(reverse_naming_dict[str(Path(file.stem).stem) + '_0000.nii.gz'])
            count += 1
    print("Done renaming {} files".format(count+1))

if __name__ == '__main__':
    task_id = "600"
    task_name = "Limbic1"
    foldername = "Task{}_{}".format(task_id, task_name)

    #setting up nnUNet folders
    # base = join(nnUNet_raw_data, foldername)
    # imagestr = join(base, "imagesTr")
    # labelstr = join(base, "labelsTr")
    # labelstr_source = join(base, 'nonConsecutive_labelsTr')

    # imagests = join(base, "imagesTs")
    # labelsts = join(base, "labelsTs")
    # labelsts_source = join(base, 'nonConsecutive_labelsTs')

    # results_nifti = "/space/freesurfer/test/nnunet/data/Results_folder/raw_output_Task600"
    # results_nifti = "/autofs/space/curv_001/users/avnish/nnUNet_output/sclimbic_vs_nnUNet_1/nnUNet_output/backup"
    imagests_mgz = "/space/freesurfer/test/nnunet/data/nnUNet_raw_data_base/nnUNet_raw_data/copy_Task600_Limbic/imagesTs/mgz"

    nnunet_labelsTs = "/autofs/space/curv_001/users/avnish/nnUNet_output/sclimbic_vs_nnUNet_1/nnUNet_output/nifti/orig_naming"
    renaming_dict_path = "/space/freesurfer/test/nnunet/data/nnUNet_raw_data_base/nnUNet_raw_data/Task600_Limbic1/renaming_key.json"
    sclimbic_orig_labels = "/autofs/space/curv_001/users/avnish/nnUNet_output/sclimbic_vs_nnUNet_1/sclimbic_output/nifti/orig_naming"
    nnunet_orig_labels = "/autofs/space/curv_001/users/avnish/nnUNet_output/sclimbic_vs_nnUNet_1/nnUNet_output/nifti/orig_naming"

    # relabel testing segmentations
    nifti_Ts_files = sorted(filter(lambda x: os.path.isfile(os.path.join(nnunet_labelsTs, x)), os.listdir(nnunet_labelsTs)))
    identifiers_Ts = np.unique([i[:-len('.nii.gz')] for i in nifti_Ts_files])
    for i in identifiers_Ts:
        convert_to_nnunet_labels(str(Path(nnunet_labelsTs + "/" + i + '.nii.gz')), str(Path(nnunet_labelsTs + "/" + i + '.nii.gz')))
    print("Segmentation labels for testing data changed sucessfully!")

    
    #!!! rename testing files for nnUnet
    # rename_files_for_task(nnunet_orig_labels, task_name=task_name, renaming_key_path=renaming_dict_path, image_or_label='label')

    # outfile = '/autofs/space/curv_001/users/avnish/nnUNet_output/sclimbic_vs_nnUNet_1/nnUNet_output/imageTs_renaming_key.json'
    # with open(outfile, 'r') as f:
    #     renaming_dict = json.load(f)
    
    # rename_files_back_to_native_names(nnunet_labelsTs, renaming_dict_path)

    # print("Successfully renamed and relabelled!")

    # labels_key = "/space/freesurfer/test/nnunet/data/nnUNet_raw_data_base/nnUNet_raw_data/Task600_Limbic1/labelsTs_renaming_key.json"
    # images_key = "/space/freesurfer/test/nnunet/data/nnUNet_raw_data_base/nnUNet_raw_data/Task600_Limbic1/imagesTs_renaming_key.json"

    # with open(labels_key, 'r') as f:
    #     labels_renaming_dict = json.load(f)
    
    # with open(images_key, 'r') as f:
    #     images_renaming_dict = json.load(f)
    
    # # renaming_dict = {}
    # # for k in labels_renaming_dict.keys():
    # #     renaming_dict[k] = [renaming_dict[k] for renaming_dict in [labels_renaming_dict, images_renaming_dict]]
    
    # renaming_dict_name = 'renaming_key.json'
    # # with open(renaming_dict_name , 'w') as f:
    # #     json.dump(renaming_dict, f, indent=2)
    

    # with open(renaming_dict_name, 'r') as f:
    #     common_keys = json.load(f)
    
    # k = list(labels_renaming_dict.keys())
    # v1 = []
    # v2 = []
    # for key in labels_renaming_dict.keys():
    #     v1.append(labels_renaming_dict[key])
    #     v2.append(common_keys[key])
    
    # res = dict(map(lambda i, j : (i,j), v1, v2))

    # label_to_image_key_name = 'labels_to_images_key.json'
    # with open(label_to_image_key_name , 'w') as f:
    #     json.dump(res, f, indent=2)