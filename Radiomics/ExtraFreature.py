import os
import yaml
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd

# 设置images和masks目录
images_dir = "path_to_images"
masks_dir = "path_to_masks"

# 设置配置文件路径
config_file = "config.yaml"

# 设置输出文件路径
output_csv = "radiomics_features.csv"

# 读取YAML配置文件
with open(config_file, 'r') as f:
    params = yaml.safe_load(f)

# 创建特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# 用于存储所有特征
all_features = []

# 遍历每对图像和掩码
for img_filename in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img_filename)
    mask_path = os.path.join(masks_dir, img_filename)  # 假设掩码文件名和图像文件名相同

    if os.path.exists(mask_path):
        image = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        features = extractor.execute(image, mask)

        # 将特征存储在字典中
        feature_dict = {key: features[key] for key in features.keys() if key not in ('diagnostics_Images', 'diagnostics_Mask')}
        feature_dict['Image'] = img_filename

        all_features.append(feature_dict)

# 使用Pandas DataFrame将所有特征保存为CSV文件
df = pd.DataFrame(all_features)
df.to_csv(output_csv, index=False)
