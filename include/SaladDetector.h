//==================================================================================================
//
//  Copyright(c)  2015  Robin Catinchi
//
//  This program is free software: you can redistribute it and/or modify it under the terms of the GNU
//	General Public License as published by the Free Software Foundation, either version 3 of the
//	License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
//  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//  See the GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License along with This program.
//  If not, see <http://www.gnu.org/licenses/>.
//
//==================================================================================================

#ifndef SALADDETECTOR_H
#define SALADDETECTOR_H

//==================================================================================================
// I N C L U D E   F I L E S   A N D   F O R W A R D   D E C L A R A T I O N S

#include "Finder.h"
#include "Histogram.h"

#include "IO/IOMVBlueFoxManager.h"

#include "HTUtility.h"
#include "CLFileSystem.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <ObjectTracker.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>

//==================================================================================================
// C O N S T A N T S

//==================================================================================================
// C L A S S E S

class SaladDetector
{
//--Methods-----------------------------------------------------------------------------------------
public:
	SaladDetector();

	~SaladDetector();

	void set_src_img( const cv::Mat& src_img );

	void set_erode( int32_t erode_size );

	void set_dilate( int32_t dilate_size );

	void init_blob_detector( bool area, bool circularity, bool color,
	                         bool convexity, bool inertia );

	void set_blob_detector_threshold( float_t min, float_t max );

	void set_blob_detector_area_range( float_t min, float_t max );

	void set_blob_detector_circularity_range( float_t min, float_t max );

	void set_blob_detector_color( uchar color );

	void set_blob_detector_convexity_range( float_t min, float_t max );

	void set_blob_detector_inertia_range( float_t min, float_t max );

	void compute();

	const std::vector< cv::Mat > get_split_src_img();

	const cv::Mat get_thres_img();

	const cv::Mat get_eroded_img();

	const cv::Mat get_dilated_img();

	const cv::Mat get_hulls_img();

	const cv::Mat get_key_points_img();

	const cv::Mat get_hulls_poly_img();

	const cv::Mat get_output_img();

private:

//--Data members------------------------------------------------------------------------------------
//private:
	cv::Mat src_img_;
	std::vector< cv::Mat > split_lab_img_;
	cv::Mat thres_img_;
	cv::Mat eroded_img_;
	cv::Mat dilated_img_;
	cv::Mat hulls_img_;
	cv::Mat key_points_img_;
	cv::Mat hulls_poly_img_;
	cv::Mat output_img_;

	cv::Mat element_erode_;
	cv::Mat element_dilate_;

	cv::SimpleBlobDetector::Params simple_blob_params_;

	uint32_t salad_count_;
	std::vector< ObjectTrackerPtr > tracked_objects_;

};

//==================================================================================================
// I N L I N E   F U N C T I O N S   C O D E   S E C T I O N

#endif //SALADDETECTOR_H
