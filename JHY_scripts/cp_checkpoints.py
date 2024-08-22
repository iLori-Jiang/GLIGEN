import os
import shutil

def backup_pth_files(source_dir, backup_dir):
    # 如果备份目录不存在，则创建它
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # 遍历源目录中的所有文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 检查文件扩展名是否为 .pth 并且不包含 _latest
            if (file.endswith('.pth') and '_latest' not in file) or file == "train_config_file.yaml":
                # 构建源文件和目标文件的完整路径
                source_file = os.path.join(root, file)
                target_file = os.path.join(backup_dir, file)
                # 复制文件到备份目录
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")


def backup_png_files(source_dir, backup_dir):
    # 如果备份目录不存在，则创建它
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # 遍历源目录中的所有文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 检查文件扩展名是否为 .png or is captions.txt
            if file.endswith('.png') or file == "captions.txt":
                # 构建源文件和目标文件的完整路径
                source_file = os.path.join(root, file)
                target_file = os.path.join(backup_dir, file)
                # 复制文件到备份目录
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")


if __name__ == "__main__":

    # ------------------------------toy_data

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/gligen_toy_data/tag01/"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/toy_dataset/checkpoint"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/toy_dataset/samples/"

    # ------------------------------ulip_chair_hed

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_chair_hed/tag01"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_chair_hed/tag01/checkpoint"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_chair_hed/tag01/samples/"

    # ------------------------------ulip_plane_canny

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_plane_canny/tag00"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny/tag00/checkpoint/"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny/tag00/samples/"

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_plane_canny/tag02"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny/tag02/checkpoint/"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny/tag02/samples/"

    # ------------------------------ulip_plane_canny_rotation

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_plane_canny_rotation/tag02"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny_rotation/tag02/checkpoint/"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny_rotation/tag02/samples/"

    # ------------------------------ulip_chair_canny_rotation
    
    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_chair_canny_rotation/tag02"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_chair_canny_rotation/tag02/checkpoint/"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_chair_canny_rotation/tag02/samples/"

    # ------------------------------ulip_chair_canny_rotation_3d

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_chair_canny_rotation_3d/tag02"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_chair_canny_rotation_3d/tag02/checkpoint/"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_chair_canny_rotation_3d/tag02/samples/"

    # ------------------------------ulip_plane_canny_rotation_3d

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_plane_canny_rotation_3d/tag02"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny_rotation_3d/tag02/checkpoint/"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane_canny_rotation_3d/tag02/samples/"

    # ------------------------------ulip_plane2chair_rotation_3d

    # source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_plane2chair_rotation_3d/tag01"
    # backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane2chair_rotation_3d/tag01/checkpoint/"
    # backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane2chair_rotation_3d/tag01/samples/"

    source_directory = "/home/haiyang/1_Repo/GLIGEN/OUTPUT/ulip_plane2chair_validate/tag01"
    backup_pth_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane2chair_validate/tag01/checkpoint/"
    backup_png_directory = "/mnt/disk2/iLori/GLIGEN/ulip_plane2chair_validate/tag01/samples/"

    # 备份 .pth 文件
    backup_pth_files(source_directory, backup_pth_directory)
    backup_png_files(source_directory, backup_png_directory)
