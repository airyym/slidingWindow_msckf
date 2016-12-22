#include "frame/frame.h"

#include "state/bodystate.h"
#include "filter/filter.h"
#include "utils/JPLquaternion.h"

#include "exceptions/general_exception.h"

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

Frame::Frame() : is_valid_(false) {
}

Frame::Frame(cv::Mat& image, cv::Ptr<cv::ORB> detector) {
    assert(image.channels() == 3 || image.channels() == 1);
    cv::Mat gray = toGray(image);

    detectKeypointsAndDescripts(detector, gray);
}

Frame::Frame(double time, const cv::Mat& image) : is_valid_(true) {
    time_ = time;
    image_ = toGray(image);
    was_processed_ = false;
}

Frame::Frame(const BodyState& body_state, std::list<Imu>::iterator hint_gyro, std::list<Imu>::iterator hint_accel)
{
    body_state_ = std::make_shared<BodyState>(body_state);
    hint_gyro_ = hint_gyro;
    hint_accel_ = hint_accel;
    camera_pose_id_ = Frame::camera_pose_counter++;
}

Frame::operator bool() const {
    return is_valid_;
}

void Frame::setIsProcessed() {
    assert(was_processed_ == false);
    was_processed_ = true;
}

bool Frame::wasProcessed() const {
    return was_processed_;
}

double Frame::getTime() const {
    return time_;
}

cv::Mat& Frame::getImage() {
    return image_;
}

const cv::Mat& Frame::cgetImage() const {
    return image_;
}

std::size_t Frame::getActiveFeaturesCount() const {
    return features_active_;
}

void Frame::setActiveFeaturesCount(std::size_t i) {
    features_active_ = i;
}

void Frame::decreaseActiveFeaturesCount(int feature_id) {
    assert(feature_ids_.find(feature_id) != std::end(feature_ids_));
    feature_ids_.erase(feature_ids_.find(feature_id));
    
    assert(features_active_ > 0);
    features_active_ -= 1;
}

BodyState& Frame::getBodyState() {
    return *body_state_;
}

const BodyState& Frame::getBodyState() const {
    return *body_state_;
}

double Frame::time() const {
    return body_state_->time();
}

const Quaternion& Frame::getBodyOrientationInGlobalFrame() const {
    return body_state_->getOrientationInGlobalFrame();
}

Quaternion Frame::getCameraOrientationInGlobalFrame(const Filter& filter) const {
    Quaternion q_C_B = filter.getBodyToCameraRotation();
    Quaternion q_B_G = getBodyOrientationInGlobalFrame();
    return q_C_B * q_B_G;
}

const Eigen::Vector3d& Frame::getBodyPositionInGlobalFrame() const {
    return body_state_->getPositionInGlobalFrame();
}

Eigen::Vector3d Frame::getCameraPositionInGlobalFrame(const Filter& filter) const {
    Quaternion q_G_C = getCameraOrientationInGlobalFrame(filter).conjugate();
    Eigen::Vector3d p_B_G = body_state_->getPositionInGlobalFrame();
    Eigen::Vector3d p_B_C = filter.getPositionOfBodyInCameraFrame();
    return p_B_G - q_G_C.toRotationMatrix() * p_B_C;
}

const Eigen::Vector3d& Frame::getBodyVelocityInGlobalFrame() const {
    return body_state_->getVelocityInGlobalFrame();
}

Quaternion Frame::getRotationToOtherPose(const CameraPose& other, const Filter& filter) const {
    Quaternion q_Cto_G = other.getCameraOrientationInGlobalFrame(filter);
    Quaternion q_Cfrom_G = getCameraOrientationInGlobalFrame(filter);
    return q_Cto_G * q_Cfrom_G.conjugate();
}

Eigen::Vector3d Frame::getPositionOfAnotherPose(const CameraPose& other, const Filter& filter) const {
    Quaternion q_Cfrom_G = getCameraOrientationInGlobalFrame(filter);
    Eigen::Matrix3d R_Cfrom_G = q_Cfrom_G.toRotationMatrix();
    Eigen::Vector3d p_Cto_G = other.getCameraPositionInGlobalFrame(filter);
    Eigen::Vector3d p_Cfrom_G = getCameraPositionInGlobalFrame(filter);
    return R_Cfrom_G * (p_Cto_G - p_Cfrom_G);
}

void Frame::rememberFeatureId(int feature_id) {
    assert(feature_ids_.find(feature_id) == std::end(feature_ids_));
    feature_ids_.insert(feature_id);
}

void Frame::updateWithStateDelta(const Eigen::VectorXd& delta_x) {
    body_state_->updateWithStateDelta(delta_x);
}

std::list<Imu>::iterator Frame::gyroHint() const {
    return hint_gyro_;
}

std::list<Imu>::iterator Frame::accelHint() const {
    return hint_accel_;
}

std::size_t Frame::getCameraPoseId() const {
    return camera_pose_id_;
}

void Frame::detectKeypointsAndDescripts(cv::Ptr< cv::ORB > detector, cv::Mat gray)
{
    detector->detectAndCompute(gray, cv::noArray(), keypoints_, descriptors_);
}

cv::Mat Frame::toGray(const cv::Mat& image) {
    switch (image.channels()) {
        case 1:
            return image;
        case 3:
        {
            cv::Mat gray;
            cv::cvtColor(image, gray, CV_BGR2GRAY);
            return gray;
        }
        default:
            throw GeneralException("Unknown image format");
    }
}

std::vector<cv::DMatch> Frame::match(cv::Ptr<cv::DescriptorMatcher> matcher, const Frame &other, float threshold) {
    if (other.descriptors_.rows == 0) {
        return std::vector<cv::DMatch>();
    }
    
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_, other.descriptors_, matches);
    
    bool use_homography_filter = true;
    if (use_homography_filter) {
        std::vector<cv::Point2f> this_pts;
        std::vector<cv::Point2f> other_pts;
        this_pts.reserve(matches.size());
        other_pts.reserve(matches.size());
        for (std::size_t i = 0; i < matches.size(); ++i) {
            this_pts.push_back(keypoints_[matches[i].queryIdx].pt);
            other_pts.push_back(other.keypoints_[matches[i].trainIdx].pt);
        }
        cv::Mat good_features_mask;
        cv::Mat H = cv::findHomography(this_pts, other_pts, CV_RANSAC, 3, good_features_mask);
        
        std::vector<cv::DMatch> good_matches;
        for (std::size_t i = 0; i < matches.size(); ++i) {
            if (good_features_mask.at<bool>(i, 0)) {
                good_matches.push_back(matches[i]);
            }
        }
        return good_matches;
    } else {
        return matches;
    }
}

double Frame::computeDistanceLimitForMatch(const std::vector<cv::DMatch>& matches) const {
    double min_distance = 100;
    double max_distance = 0;
    double mean_distance = 0;
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const cv::DMatch& match = matches[i];
        mean_distance += match.distance;
        if (match.distance < min_distance) {
            min_distance = match.distance;
        }
        if (match.distance > max_distance) {
            max_distance = match.distance;
        }
    }
    mean_distance /= matches.size();
    return std::max(2*min_distance, 5.0);
}