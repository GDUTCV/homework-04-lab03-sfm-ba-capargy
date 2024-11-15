import numpy as np
import cv2


def compute_ba_residuals(parameters: np.ndarray, intrinsics: np.ndarray, num_cameras: int, points2d: np.ndarray,
                         camera_idxs: np.ndarray, points3d_idxs: np.ndarray) -> np.ndarray:
    """
    For each point2d in <points2d>, find its 3d point, reproject it back into the image and return the residual
    i.e. euclidean distance between the point2d and reprojected point.

    Args:
        parameters: list of camera parameters [r1, r2, r3, t1, t2, t3, ...] where r1, r2, r3 corresponds to the
                    Rodriguez vector. There are 6C + 3M parameters where C is the number of cameras
        intrinsics: camera intrinsics 3 x 3 array
        num_cameras: number of cameras, C
        points2d: N x 2 array of 2d points
        camera_idxs: camera_idxs[i] returns the index of the camera for points2d[i]
        points3d_idxs: points3d[points3d_idxs[i]] returns the 3d point corresponding to points2d[i]

    Returns:
        N residuals

    """
    num_camera_parameters = 6 * num_cameras #设置张量的大小
    camera_parameters = parameters[:num_camera_parameters] #提取6C
    points3d = parameters[num_camera_parameters:] #提取3M
    num_points3d = points3d.shape[0] // 3 
    points3d = points3d.reshape(num_points3d, 3) #相机3D坐标张量形状

    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    camera_rvecs = camera_parameters[:, :3] #3平移
    camera_tvecs = camera_parameters[:, 3:] #3旋转

    extrinsics = []
    for rvec in camera_rvecs:
        rot_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics.append(rot_mtx)
    extrinsics = np.array(extrinsics)  # num_cameras x 3 x 3
    extrinsics = np.concatenate([extrinsics, camera_tvecs.reshape(-1, 3, 1)], axis=2)  # num_cameras x 3 x 4

    residuals = np.zeros(shape=points2d.shape[0], dtype=float)
    """ 
    YOUR CODE HERE: 
    NOTE: DO NOT USE LOOPS 
    HINT: I used np.matmul; np.sum; np.sqrt; np.square, np.concatenate etc.
    """
    selected_points3d = points3d[points3d_idxs] #N3，选取相应的相机的内点内点个数所以是N

    homo_3d_points = np.concatenate((selected_points3d, np.ones((selected_points3d.shape[0], 1))), axis=1) #N3N4
    homo_3d_points_T = np.transpose(homo_3d_points) #N4 4N

    selected_extrinsics = extrinsics[camera_idxs] # N x 3 x 4，调用不同camera_idxs相应的内点
    P = np.matmul(intrinsics, selected_extrinsics) # 33*N34 N34

    calculated_points3d = np.einsum('ijk,ki->ij', P, homo_3d_points_T) #N34 4N 3N；这个是ik,ki→i（只取乘积的对角线组成向量），p3D有j个（N个点）这类向量
    calculated_points3d /= calculated_points3d[:, -1].reshape((calculated_points3d.shape[0], 1)) #归一化
    calculated_points2d = calculated_points3d[:, :-1] #去Z轴

    residuals = np.linalg.norm(points2d - calculated_points2d, axis=1)
    """ END YOUR CODE HERE """
    return residuals
