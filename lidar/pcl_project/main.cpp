#include <pcl/io/pcd_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

float computeAdaptiveThreshold(float distance, float angleResolution, float noise) {
    return distance * std::sin(angleResolution) + noise;
}

pcl::visualization::PCLVisualizer::Ptr visualizeClusters(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
    std::vector<pcl::PointIndices>& cluster_indices) {

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Clusters Viewer"));

    int cluster_id = 0;
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : indices.indices) {
            cluster->points.push_back(cloud->points[idx]);
        }

        std::string name = "cluster_" + std::to_string(cluster_id);
        viewer->addPointCloud(cluster, name);
        cluster_id++;
    }

    viewer->spin();
    return viewer;
}

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Load KITTI binary file
    std::ifstream inputFile("/home/raja/Desktop/lidar/pcl_project/kitti_sample.bin", std::ios::binary);
    if (!inputFile) {
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }

    // Load points from binary file
    while (inputFile.good()) {
        float x, y, z, intensity;
        inputFile.read(reinterpret_cast<char*>(&x), sizeof(float));
        inputFile.read(reinterpret_cast<char*>(&y), sizeof(float));
        inputFile.read(reinterpret_cast<char*>(&z), sizeof(float));
        inputFile.read(reinterpret_cast<char*>(&intensity), sizeof(float));

        cloud->points.emplace_back(x, y, z);
    }

    inputFile.close();
    cloud->width = cloud->points.size();
    cloud->height = 1;

    std::cout << "Loaded " << cloud->size() << " points from KITTI file." << std::endl;

    // KdTree for nearest neighbor search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    float angleResolution = M_PI / 180; // 1-degree angular resolution
    float noise = 0.05; // Measurement noise

    // Dynamic clustering threshold
    for (const auto& point : cloud->points) {
        float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        float threshold = computeAdaptiveThreshold(distance, angleResolution, noise);
        ec.setClusterTolerance(threshold);
    }

    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::cout << "Clusters found: " << cluster_indices.size() << std::endl;

    // Visualize the clusters
    visualizeClusters(cloud, cluster_indices);

    return 0;
}

