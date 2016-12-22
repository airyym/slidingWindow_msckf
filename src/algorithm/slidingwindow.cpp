#include "algorithm/slidingwindow.h"

#include "frame/frame.h"

#include <boost/circular_buffer.hpp>
#include <stdexcept>
#include <utility>

SlidingWindow::SlidingWindow(int max_camera_poses)
: buffer_(max_camera_poses) {
}

Frame& SlidingWindow::operator[](std::size_t i) {
    return buffer_[i];
}

const Frame& SlidingWindow::operator[](std::size_t i) const {
    return buffer_[i];
}

void SlidingWindow::deleteOldestCameraPose() {
    if (buffer_.empty()) {
        throw std::runtime_error("Trying to delete camera pose from empty SlidingWindow");
    }
    buffer_.pop_front();
}

void SlidingWindow::addNewCameraPose(CameraPose&& pose) {
    if (buffer_.full()) {
        throw std::runtime_error("SlidingWindow is full. Cannot add another CameraPose.");
    }
    buffer_.push_back(std::move(pose));
}

SlidingWindow::iterator SlidingWindow::begin() {
    return buffer_.begin();
}

SlidingWindow::iterator SlidingWindow::end() {
    return buffer_.end();
}

SlidingWindow::const_iterator SlidingWindow::begin() const {
    return buffer_.begin();
}

SlidingWindow::const_iterator SlidingWindow::end() const {
    return buffer_.end();
}

SlidingWindow::reverse_iterator SlidingWindow::rbegin() noexcept {
    return buffer_.rbegin();
}

SlidingWindow::reverse_iterator SlidingWindow::rend() noexcept {
    return buffer_.rend();
}

SlidingWindow::const_reverse_iterator SlidingWindow::rbegin() const noexcept {
    return buffer_.rbegin();
}

SlidingWindow::const_reverse_iterator SlidingWindow::rend() const noexcept {
    return buffer_.rend();
}

Frame& SlidingWindow::front() {
    if (buffer_.empty()) {
        throw std::runtime_error("SlidingWindow is empty. Cannot return front element.");
    }
    return buffer_.front();
}

Frame& SlidingWindow::back() {
    if (buffer_.empty()) {
        throw std::runtime_error("SlidingWindow is empty. Cannot return back element.");
    }
    return buffer_.back();
}

std::size_t SlidingWindow::size() const {
    return buffer_.size();
}

bool SlidingWindow::empty() const {
    return buffer_.empty();
}