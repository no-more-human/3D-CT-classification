import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np

def force_pydicom_read(dcm_folder, output_path):
    """保底方案：当 SimpleITK 无法识别某些不规范序列时，用 pydicom 强制读取并拼接"""
    dcm_files = glob.glob(os.path.join(dcm_folder, "*"))
    dcm_files = [f for f in dcm_files if os.path.isfile(f)]
    
    slices = []
    for f in dcm_files:
        try:
            slices.append(pydicom.dcmread(f))
        except:
            continue
            
    if len(slices) == 0:
        raise ValueError(f"该文件夹内没有有效的 DICOM 文件: {dcm_folder}")
        
    # 按照切片位置（ImagePositionPatient）或者实例编号排序
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass
            
    volume = np.stack([s.pixel_array for s in slices], axis=0) # (D, H, W)
    sitk_img = sitk.GetImageFromArray(volume.astype(np.int16))
    
    try:
        sitk_img.SetSpacing((float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[0].SliceThickness)))
    except:
        pass
        
    sitk.WriteImage(sitk_img, output_path)


def dcm2nii(dcm_folder, output_path):
    """核心转换逻辑"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_folder)
    
    if not dicom_names:
        # 如果标准读取器找不到序列，启动 pydicom 强制修复
        print("-> 提示: 标准读取器未识别到序列，正在启动 pydicom 强行打包...")
        force_pydicom_read(dcm_folder, output_path)
        return

    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)


def process_all_data(base_dir, output_base_dir):
    categories = ['FD', 'OF']
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        out_cat_dir = os.path.join(output_base_dir, cat)
        os.makedirs(out_cat_dir, exist_ok=True)
        
        if not os.path.exists(cat_dir):
            continue
            
        for patient_id in os.listdir(cat_dir):
            patient_dir = os.path.join(cat_dir, patient_id)
            if not os.path.isdir(patient_dir): 
                continue
            
            # 获取 1-20 文件夹下的子目录 (如 3DFD 或 3DOF)
            dcm_folder = None
            for sub in os.listdir(patient_dir):
                sub_path = os.path.join(patient_dir, sub)
                if os.path.isdir(sub_path):
                    dcm_folder = sub_path
                    break
            
            if dcm_folder:
                out_file = os.path.join(out_cat_dir, f"{cat}_{patient_id}.nii.gz")
                print(f"正在处理: {cat} 样本 {patient_id}...")
                try:
                    dcm2nii(dcm_folder, out_file)
                    print(f"成功保存 -> {out_file}")
                except Exception as e:
                    print(f"[ERROR] 错误: 无法转换 {dcm_folder}, 原因: {e}")

if __name__ == "__main__":
    # 修改为你的全英文路径
    raw_data_path = r"F:\CT_Dataset"
    processed_data_path = r"F:\python\3DCT_Classification\dataset\NIfTI_Data" 
    
    print("开始执行 3D 医疗影像 NIfTI 转换流程...")
    process_all_data(raw_data_path, processed_data_path)
    print("\n转换任务全部结束！请检查 dataset 文件夹。")