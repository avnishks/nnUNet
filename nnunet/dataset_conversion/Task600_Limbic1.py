# from ntpath import join
from fileinput import filename
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import subprocess
from pathlib import Path

import SimpleITK as sitk
from nnunet.paths import nnUNet_raw_data
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.utilities.sitk_stuff import copy_geometry

def convert_labels_to_nnunet(source_nifti: str, target_nifti: str):
    img = sitk.ReadImage(source_nifti)
    img_npy = sitk.GetArrayFromImage(img)
    nnunet_seg = np.zeros(img_npy.shape, dtype=np.uint8) #[0-255]!!!

    nnunet_seg[img_npy == 26] = 1    # Left-Nucleus-Accumbens
    nnunet_seg[img_npy == 58] = 2    # Right-Nucleus-Accumbens
    nnunet_seg[img_npy == 819] = 3   # Left-HypoThal-noM
    nnunet_seg[img_npy == 820] = 4   # Right-HypoThal-noMB
    nnunet_seg[img_npy == 821] = 5   # Left-Fornix
    nnunet_seg[img_npy == 822] = 6   # Right-Fornix
    nnunet_seg[img_npy == 843] = 7   # Left-MammillaryBody
    nnunet_seg[img_npy == 844] = 8   # Right-MammillaryBody
    nnunet_seg[img_npy == 853] = 9   # Mid-AntCom
    nnunet_seg[img_npy == 865] = 10  # Left-Basal-Forebrain
    nnunet_seg[img_npy == 866] = 11  # Right-Basal-Forebrain
    nnunet_seg[img_npy == 869] = 12  # Left-SeptalNuc
    nnunet_seg[img_npy == 870] = 13  # Right-SeptalNuc

    nnunet_seg_itk = sitk.GetImageFromArray(nnunet_seg)
    nnunet_seg_itk = copy_geometry(nnunet_seg_itk, img)
    sitk.WriteImage(nnunet_seg_itk, target_nifti)

def convert_labels_back_to_limbic(source_nifti: str, target_nifti: str):
    nnunet_itk = sitk.ReadImage(source_nifti)
    nnunet_npy = sitk.GetArrayFromImage(nnunet_itk)
    limbic_seg = np.zeros(nnunet_npy.shape, dtype=np.uint8)

    limbic_seg[nnunet_npy == 1] = 26   # Left-Nucleus-Accumbens
    limbic_seg[nnunet_npy == 2] = 58   # Right-Nucleus-Accumbens
    limbic_seg[nnunet_npy == 3] = 819  # Left-HypoThal-noM
    limbic_seg[nnunet_npy == 4] = 820  # Right-HypoThal-noMB
    limbic_seg[nnunet_npy == 5] = 821  # Left-Fornix
    limbic_seg[nnunet_npy == 6] = 822  # Right-Fornix
    limbic_seg[nnunet_npy == 7] = 843  # Left-MammillaryBody
    limbic_seg[nnunet_npy == 8] = 844  # Right-MammillaryBody
    limbic_seg[nnunet_npy == 9] = 853  # Mid-AntCom
    limbic_seg[nnunet_npy == 10] = 865 # Left-Basal-Forebrain
    limbic_seg[nnunet_npy == 11] = 866 # Right-Basal-Forebrain
    limbic_seg[nnunet_npy == 12] = 869 # Left-SeptalNuc
    limbic_seg[nnunet_npy == 13] = 870 # Right-SeptalNuc
    limbic_seg_itk = sitk.GetImageFromArray(limbic_seg)
    limbic_seg_itk = copy_geometry(limbic_seg_itk, nnunet_itk)
    sitk.WriteImage(limbic_seg_itk, target_nifti)

def remap_sclimbic_to_nnunet(source_nifti: str, target_nifti: str):
    img = sitk.ReadImage(source_nifti)
    img_npy = sitk.GetArrayFromImage(img)
    nnunet_seg = np.zeros(img_npy.shape, dtype=np.uint8) #[0-255]!!!

    nnunet_seg[img_npy == 26] =26    # Left-Nucleus-Accumbens
    nnunet_seg[img_npy == 58] = 58    # Right-Nucleus-Accumbens
    nnunet_seg[img_npy == 819] = 51   # Left-HypoThal-noM
    nnunet_seg[img_npy == 820] = 52   # Right-HypoThal-noMB
    nnunet_seg[img_npy == 821] = 53   # Left-Fornix
    nnunet_seg[img_npy == 822] = 54   # Right-Fornix
    nnunet_seg[img_npy == 843] = 75   # Left-MammillaryBody
    nnunet_seg[img_npy == 844] = 76   # Right-MammillaryBody
    nnunet_seg[img_npy == 853] = 85   # Mid-AntCom
    nnunet_seg[img_npy == 865] = 97  # Left-Basal-Forebrain
    nnunet_seg[img_npy == 866] = 98  # Right-Basal-Forebrain
    nnunet_seg[img_npy == 869] = 101  # Left-SeptalNuc
    nnunet_seg[img_npy == 870] = 102  # Right-SeptalNuc

    nnunet_seg_itk = sitk.GetImageFromArray(nnunet_seg)
    nnunet_seg_itk = copy_geometry(nnunet_seg_itk, img)
    sitk.WriteImage(nnunet_seg_itk, target_nifti)

# def rename_files_for_task(folder, task_name):
#     pathiter = (os.path.join(root, filename)
#         for root, _, filenames in os.walk(folder)
#         for filename in filenames    
#     )

#     for path in pathiter:
#         newname = path.replace('limbic', task_name)
#         if newname != path:
#             os.rename(path, newname)
#     print("Done renaming file!")

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
                if renaming_key.__contains__(str(Path(file.name))):
                    rename_idx = renaming_key[str(Path(file.name))].split('_')[1].split('.')[0]
                    new_name = str(file.parent) + '/' + task_name + f'_{rename_idx}' + '.nii.gz'
                    file.replace(new_name)
                    count += 1
            print(f"Renamed {count} files.")
    else:
        print ("Not implemented error!")

    return renaming_key

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
    base = join(nnUNet_raw_data, foldername)
    imagestr = join(base, "imagesTr")
    labelstr = join(base, "labelsTr")
    # labelstr_source = join(base, 'nonConsecutive_labelsTr')

    imagests = join(base, "imagesTs")
    labelsts = join(base, "labelsTs")
    # labelsts_source = join(base, 'nonConsecutive_labelsTs')


    # # rename training files for nnUnet
    # rename_files_for_task(imagestr, task_name)
    # rename_files_for_task(labelstr, task_name)

    # # rename testing files for nnUnet
    image_outfile = str(Path(base + '/imagesTs_renaming_key.json'))
    label_outfile = str(Path(base + '/labelsTs_renaming_key.json'))
    labelsTs_renaming_key_path = "/space/freesurfer/test/nnunet/data/nnUNet_raw_data_base/nnUNet_raw_data/Task600_Limbic1/imagesTs_renaming_key.json"

    # imagesTs_renaming_key = rename_files_for_task(imagests, task_name, image_or_label='image', renaming_key={})
    # with open(image_outfile, 'w') as f:
    #     json.dump(imagesTs_renaming_key, f, indent=2)
    
    # labelsTs_renaming_key = rename_files_for_task(labelsts, task_name, image_or_label='label', renaming_key={})
    # with open(label_outfile, 'w') as f:
    #     json.dump(labelsTs_renaming_key, f, indent=2)

    rename_key = rename_files_for_task(labelsts, task_name, image_or_label='label', renaming_key_path=labelsTs_renaming_key_path)

    # #relabel training segmentations
    # nii_files = nifti_files(imagestr, join=False)
    # identifiers = np.unique([i[:-len('_0000.nii.gz')] for i in nii_files])
    # for i in identifiers:
    #     convert_labels_to_nnunet(join(labelstr_source, i + '.nii.gz'), join(labelstr, i + '.nii.gz'))
    # print("Segmentation labels for training data changed sucessfully!")

    # relabel testing segmentations
    # nii_files = nifti_files(imagests, join=False)
    # identifiers_Ts = np.unique([i[:-len('_0000.nii.gz')] for i in nii_files])
    # for i in identifiers_Ts:
    #     remap_sclimbic_to_nnunet(join(labelsts, i + '.nii.gz'), join(labelsts, i + '.nii.gz'))
    # print("Segmentation labels for testing data changed sucessfully!")

    # rename to nnUNet
    # predict_files="/autofs/space/curv_001/users/avnish/nnUNet_output/sclimbic_vs_nnUNet_1/nnUNet_output/predict"
    # nnunet_predict_labelsTs_renaming_key = rename_files_for_task(predict_files, task_name, image_or_label='label', renaming_key={})
    # with open(label_outfile, 'w') as f:
    #     json.dump(nnunet_predict_labelsTs_renaming_key, f, indent=2)

    # generate_dataset_json(join(base, 'dataset.json'),
    #                       imagestr,
    #                       imagests,
    #                       ("MRI",),
    #                       {
    #                           0: 'background',
    #                           1: "Left-Nucleus-Accumbens",
    #                           2: "Right-Nucleus-Accumbens",
    #                           3: "Left-HypoThal-noMB",
    #                           4: "Right-HypoThal-noMB",
    #                           5: "Left-Fornix",
    #                           6: "Right-Fornix",
    #                           7: "Left-MammillaryBody",
    #                           8: "Right-MammillaryBody",
    #                           9: "Mid-AntCom",
    #                           10: "Left-Basal-Forebrain",
    #                           11: "Right-Basal-Forebrain",
    #                           12: "Left-SeptalNuc",
    #                           13: "Right-SeptalNuc"
    #                       },
    #                       task_name,
    #                       license='<license>',
    #                       dataset_description='<description>',
    #                       dataset_reference='<dataset reference>',
    #                       dataset_release='<dataset release>')
    
    # TO-DO: Convert labels back to limbic after inference.

    # print("Dataset.json created successfully!")
