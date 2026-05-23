# from PIL import Image
# import torch
# import cv2
# from romatch import roma_outdoor
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.backends.mps.is_available():
#     device = torch.device('mps')
#
# if __name__ == "__main__":
#     from argparse import ArgumentParser
#     parser = ArgumentParser()
#     parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
#     parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
#
#     args, _ = parser.parse_known_args()
#     im1_path = args.im_A_path
#     im2_path = args.im_B_path
#
#     # Create model
#     roma_model = roma_outdoor(device=device)
#
#
#     W_A, H_A = Image.open(im1_path).size
#     W_B, H_B = Image.open(im2_path).size
#
#     # Match
#     warp, certainty = roma_model.match(im1_path, im2_path, device=device)
#     # Sample matches for estimation
#     matches, certainty = roma_model.sample(warp, certainty)
#     kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
#     F, mask = cv2.findFundamentalMat(
#         kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
#     )
from PIL import Image
import torch
import cv2
import numpy as np  # 新增：用于图像处理
from romatch import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    parser.add_argument("--output_path", default="fundamental_matches.png", type=str)  # 新增：输出图片路径
    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # 创建模型
    roma_model = roma_outdoor(device=device)

    # 读取图像尺寸
    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # 特征匹配
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(),
        ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC,
        confidence=0.999999, maxIters=10000
    )

    # 新增：可视化匹配结果并保存
    # 读取原图
    img1 = cv2.imread(im1_path)
    img2 = cv2.imread(im2_path)
    # 调整图像尺寸一致（方便拼接）
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_height = max(h1, h2)
    img1_pad = cv2.copyMakeBorder(img1, 0, max_height - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    img2_pad = cv2.copyMakeBorder(img2, 0, max_height - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    # 拼接图像
    combined = np.hstack((img1_pad, img2_pad))
    # 绘制匹配线（只画RANSAC筛选后的有效匹配）
    valid_kpts1 = kpts1[mask.ravel() == 1].cpu().numpy()
    valid_kpts2 = kpts2[mask.ravel() == 1].cpu().numpy()
    for (x1, y1), (x2, y2) in zip(valid_kpts1, valid_kpts2):
        cv2.line(combined, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (0, 255, 0), 1)  # 绿色匹配线
        cv2.circle(combined, (int(x1), int(y1)), 3, (0, 0, 255), -1)  # 红色关键点（左图）
        cv2.circle(combined, (int(x2) + w1, int(y2)), 3, (255, 0, 0), -1)  # 蓝色关键点（右图）
    # 保存结果
    cv2.imwrite(args.output_path, combined)
    print(f"匹配结果已保存至：{args.output_path}")