#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H

#include "frame/frame.h"

#include <boost/circular_buffer.hpp>
#include <utility>

typedef boost::circular_buffer<CameraPose>::iterator iterator;
typedef boost::circular_buffer<CameraPose>::const_iterator const_iterator;
typedef boost::circular_buffer<CameraPose>::reverse_iterator reverse_iterator;
typedef boost::circular_buffer<CameraPose>::const_reverse_iterator const_reverse_iterator;

/**
 * @brief Sliding window, container of frames.
 * 
 * @author Doom
 */

class SlidingWindow {
public:  
    SlidingWindow(int max_camera_poses);
    
    CameraPose& operator[](std::size_t i);
    const CameraPose& operator[](std::size_t i) const;
    
    void deleteOldestCameraPose();
    void addNewCameraPose(CameraPose&& pose);
    
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    
    reverse_iterator rbegin() noexcept;
    reverse_iterator rend() noexcept;
    const_reverse_iterator rbegin() const noexcept;
    const_reverse_iterator rend() const noexcept;
    
    Frame& front();
    Frame& back();
        
    std::size_t size() const;
    bool empty() const;
    
private:
    boost::circular_buffer<Frame> buffer_;
};

#endif //SLIDINGWINDOW_H