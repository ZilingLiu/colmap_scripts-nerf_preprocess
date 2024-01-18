import open3d as o3d

# 加载点云数据
pcd = o3d.io.read_point_cloud("milk.ply")

# 可视化点云数据
o3d.visualization.draw_geometries([pcd])