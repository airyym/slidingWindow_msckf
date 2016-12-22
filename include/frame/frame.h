#ifndef FRAME_H
#define FRAME_H

#include "imu/sensorflow.h"
#include "utils/JPLquaternion.h"

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <memory>
#include <set>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

class BodyState;

/**
 * @brief Frame class, include its keypoints, pose and so on.
 * 
 * @author Doom
 */
class Frame {
public:
    /** @brief default construction method */
    Frame();
    
    /** @brief used in feature tracker */
    Frame(cv::Mat& image, cv::Ptr<cv::ORB> detector);
    
    /** @brief used in filter */
    Frame(double time, const cv::Mat& image);
    
    /** @brief used in propagation and update */
    Frame(const BodyState& body_state, std::list<Imu>::iterator hint_gyro, std::list<Imu>::iterator hint_accel);
    
    Frame(const Frame& other) = default;
    
    Frame& operator=(const Frame& oher) = default;
    
    operator bool() const;
    
    std::vector<cv::DMatch> match(cv::Ptr<cv::DescriptorMatcher> matcher, const Feature& other,
            float threshold = 0.5);
    
    std::vector<cv::KeyPoint>& keypoints();
    const std::vector<cv::KeyPoint>& keypoints() const;
    
    double computeDistanceLimitForMatch(const std::vector<cv::DMatch>& matches) const;
    
    void setIsProcessed();
    bool wasProcessed() const;

    double getTime() const;
    cv::Mat& getImage();
    const cv::Mat& cgetImage() const;
    
    std::size_t getActiveFeaturesCount() const;
    void setActiveFeaturesCount(std::size_t i);
    void decreaseActiveFeaturesCount(int feature_id);
    
    BodyState& getBodyState();
    const BodyState& getBodyState() const;
    
    /**
     * @brief Time of body state
     */
    double time() const;
    
    /** @brief Get orientation \f$ {}^G \mathbf{q}_B \f$ of this camera frame in global frame. */
    const Quaternion& getBodyOrientationInGlobalFrame() const;
    
    /** @brief Get orientation \f$ {}^G \mathbf{q}_C \f$ of this camera frame in global frame. */
    Quaternion getCameraOrientationInGlobalFrame(const Filter& filter) const;
    
    /** @brief Get position \f$ {}^G \mathbf{p}_B \f$ of this camera frame in global frame. */
    const Eigen::Vector3d& getBodyPositionInGlobalFrame() const;
    
    /** @brief Get position \f$ {}^G \mathbf{p}_C \f$ of this camera frame in global frame. */
    Eigen::Vector3d getCameraPositionInGlobalFrame(const Filter& filter) const;
    
    /** @brief Get velocity \f$ {}^G\mathbf{v}_B \f$ of this camera frame in global frame. */
    const Eigen::Vector3d& getBodyVelocityInGlobalFrame() const;
    
    Quaternion getRotationToOtherPose(const CameraPose& other, const Filter& filter) const;
    
    Eigen::Vector3d getPositionOfAnotherPose(const CameraPose& other, const Filter& filter) const;

    void rememberFeatureId(int feature_id);

    void updateWithStateDelta(const Eigen::VectorXd& delta_x);
    
    ImuBuffer::iterator gyroHint() const;
    ImuBuffer::iterator accelHint() const;
    
    std::size_t getCameraPoseId() const;

protected:
    bool is_valid_;
    double time_;
    cv::Mat image_;
    bool was_processed_;
    
    static std::size_t camera_pose_counter;
    std::size_t camera_pose_id_;
    std::set<int> feature_ids_;
    std::size_t features_active_;
    std::shared_ptr<BodyState> body_state_;
    ImuBuffer::iterator hint_gyro_;
    ImuBuffer::iterator hint_accel_;
  
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    void detectKeypointsAndDescripts(cv::Ptr<cv::ORB> detector, cv::Mat gray);

    static cv::Mat toGray(const cv::Mat& image);
};

#endif //FRAME_H